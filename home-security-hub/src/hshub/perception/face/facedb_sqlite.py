"""SQLite-backed face embedding storage."""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime
from datetime import timezone

import numpy as np

from hshub.types import MatchResult

LOGGER = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return vec
    return vec / norm


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class FaceDBIndex:
    SQL_CREATE_PERSONS = """
        CREATE TABLE IF NOT EXISTS persons (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL
        )
    """
    SQL_SELECT_ALL = "SELECT person_id, name, embedding FROM persons ORDER BY person_id"
    SQL_INSERT_PERSON = (
        "INSERT INTO persons (name, embedding, created_at, last_seen) VALUES (?, ?, ?, ?)"
    )
    SQL_UPDATE_NAME = "UPDATE persons SET name = ? WHERE person_id = ?"
    SQL_UPDATE_PERSON = "UPDATE persons SET embedding = ?, last_seen = ? WHERE person_id = ?"

    def __init__(
        self,
        db_path: str,
        match_threshold: float,
        margin: float,
        commit_every_n: int,
        commit_every_sec: float,
        centroid_alpha: float,
    ) -> None:
        self.db_path = db_path
        self.match_threshold = float(match_threshold)
        self.margin = float(margin)
        self.commit_every_n = max(1, int(commit_every_n))
        self.commit_every_sec = max(0.1, float(commit_every_sec))
        self.centroid_alpha = float(np.clip(centroid_alpha, 0.0, 1.0))

        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA temp_store=MEMORY")

        self.conn.execute(self.SQL_CREATE_PERSONS)
        self.conn.commit()

        self.person_ids: np.ndarray = np.empty((0,), dtype=np.int64)
        self.names: list[str] = []
        self.embedding_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.id_to_index: dict[int, int] = {}

        self.pending_ops = 0
        self.last_commit_monotonic = time.monotonic()
        self.pending_updates: dict[int, tuple[np.ndarray, str]] = {}

        self._load_people()

    def _load_people(self) -> None:
        rows = self.conn.execute(self.SQL_SELECT_ALL).fetchall()
        if not rows:
            self.person_ids = np.empty((0,), dtype=np.int64)
            self.names = []
            self.embedding_matrix = np.empty((0, 0), dtype=np.float32)
            self.id_to_index = {}
            return

        person_ids: list[int] = []
        names: list[str] = []
        vectors: list[np.ndarray] = []
        expected_dim: int | None = None

        for person_id, name, blob in rows:
            vec = normalize_embedding(blob_to_embedding(blob))
            if vec.size == 0:
                continue
            if expected_dim is None:
                expected_dim = int(vec.size)
            if vec.size != expected_dim:
                LOGGER.warning(
                    "Skipping person_id=%s: embedding dim %s != %s",
                    person_id,
                    vec.size,
                    expected_dim,
                )
                continue
            person_ids.append(int(person_id))
            names.append(str(name))
            vectors.append(vec)

        if not vectors:
            self.person_ids = np.empty((0,), dtype=np.int64)
            self.names = []
            self.embedding_matrix = np.empty((0, 0), dtype=np.float32)
            self.id_to_index = {}
            return

        self.person_ids = np.asarray(person_ids, dtype=np.int64)
        self.names = names
        self.embedding_matrix = np.vstack(vectors).astype(np.float32, copy=False)
        self.id_to_index = {int(pid): idx for idx, pid in enumerate(self.person_ids)}

    def _flush_if_needed(self, force: bool = False) -> None:
        elapsed = time.monotonic() - self.last_commit_monotonic
        should_flush = force or self.pending_ops >= self.commit_every_n
        if not should_flush and self.pending_ops > 0 and elapsed >= self.commit_every_sec:
            should_flush = True
        if not should_flush:
            return

        if self.pending_updates:
            rows = [
                (embedding_to_blob(embedding), ts, person_id)
                for person_id, (embedding, ts) in self.pending_updates.items()
            ]
            self.conn.executemany(self.SQL_UPDATE_PERSON, rows)
            self.pending_updates.clear()

        if self.pending_ops > 0:
            self.conn.commit()
            self.pending_ops = 0
            self.last_commit_monotonic = time.monotonic()

    def flush(self) -> None:
        self._flush_if_needed(force=True)

    def maybe_flush(self) -> None:
        self._flush_if_needed(force=False)

    def close(self) -> None:
        self.flush()
        self.conn.close()

    def _append_person_cache(self, person_id: int, name: str, embedding: np.ndarray) -> None:
        normalized = normalize_embedding(embedding)
        if self.embedding_matrix.size == 0:
            self.embedding_matrix = normalized.reshape(1, -1)
        else:
            if normalized.shape[0] != self.embedding_matrix.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: {normalized.shape[0]} != "
                    f"{self.embedding_matrix.shape[1]}"
                )
            self.embedding_matrix = np.vstack((self.embedding_matrix, normalized))

        self.person_ids = np.append(self.person_ids, np.int64(person_id))
        self.names.append(name)
        self.id_to_index[int(person_id)] = len(self.names) - 1

    def find_best_match(self, face_embedding: np.ndarray) -> MatchResult | None:
        if self.embedding_matrix.size == 0:
            return None

        query = normalize_embedding(face_embedding)
        similarities = self.embedding_matrix @ query
        if similarities.size == 0:
            return None

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if similarities.size > 1:
            top2 = np.sort(np.partition(similarities, -2)[-2:])
            second_score = float(top2[0])
        else:
            second_score = -1.0

        if best_score < self.match_threshold:
            return None
        if similarities.size > 1 and (best_score - second_score) < self.margin:
            return None

        person_id = int(self.person_ids[best_idx])
        return MatchResult(
            person_id=person_id,
            name=self.names[best_idx],
            best_score=best_score,
            second_score=second_score,
        )

    def insert_new_person(self, embedding: np.ndarray) -> MatchResult:
        norm_embedding = normalize_embedding(embedding)
        ts = now_iso()
        cursor = self.conn.execute(
            self.SQL_INSERT_PERSON,
            ("unknown", embedding_to_blob(norm_embedding), ts, ts),
        )
        person_id = int(cursor.lastrowid)
        name = f"person_{person_id}"
        self.conn.execute(self.SQL_UPDATE_NAME, (name, person_id))

        self.pending_ops += 2
        self._append_person_cache(person_id=person_id, name=name, embedding=norm_embedding)
        self._flush_if_needed()

        return MatchResult(
            person_id=person_id,
            name=name,
            best_score=1.0,
            second_score=-1.0,
        )

    def update_seen(self, person_id: int, embedding: np.ndarray) -> None:
        idx = self.id_to_index.get(person_id)
        if idx is None:
            return

        query = normalize_embedding(embedding)
        current = self.embedding_matrix[idx]
        updated = normalize_embedding(
            ((1.0 - self.centroid_alpha) * current) + (self.centroid_alpha * query)
        )
        self.embedding_matrix[idx] = updated

        self.pending_updates[person_id] = (updated, now_iso())
        self.pending_ops += 1
        self._flush_if_needed()
