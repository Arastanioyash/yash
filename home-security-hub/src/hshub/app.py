from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from typing import Any

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

# -----------------------------
# Key configuration parameters
# -----------------------------
DB_PATH = "faces.db"
VIDEO_PATH = "test1.mp4"
MATCH_THRESHOLD = 0.50
MARGIN = 0.04
COMMIT_EVERY_N = 50
COMMIT_EVERY_SEC = 1.0
DETECT_EVERY_N = 1
DETECT_FRAME_SCALE = 1.0
DET_SIZE = (640, 640)
DETECTION_SCORE_THRESHOLD = 0.60
MIN_FACE_SIZE_PX = 30
MAX_FACE_ASPECT_RATIO = 1.6
MIN_FACE_AREA_RATIO = 0.002
CENTROID_ALPHA = 0.05
NEW_PERSON_SAVE_DELAY_SEC = 2.0
PENDING_NEW_SIM_THRESHOLD = 0.60
PENDING_NEW_STALE_SEC = 2.0
DISPLAY_W, DISPLAY_H = 1280, 720


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


@dataclass(frozen=True)
class MatchResult:
    person_id: int
    name: str
    best_score: float
    second_score: float


@dataclass
class PendingCandidate:
    embedding: np.ndarray
    first_seen: float
    last_seen: float


class NewPersonGate:
    def __init__(self, delay_sec: float, similarity_threshold: float, stale_sec: float) -> None:
        self.delay_sec = max(0.0, delay_sec)
        self.similarity_threshold = float(np.clip(similarity_threshold, 0.0, 1.0))
        self.stale_sec = max(0.1, stale_sec)
        self.candidates: list[PendingCandidate] = []

    def _prune_stale(self, now_monotonic: float) -> None:
        self.candidates = [
            candidate
            for candidate in self.candidates
            if (now_monotonic - candidate.last_seen) <= self.stale_sec
        ]

    def tick(self, now_monotonic: float) -> None:
        self._prune_stale(now_monotonic)

    def discard_if_similar(self, embedding: np.ndarray, now_monotonic: float) -> None:
        if not self.candidates:
            return
        self._prune_stale(now_monotonic)
        if not self.candidates:
            return

        query = normalize_embedding(embedding)
        matrix = np.vstack([candidate.embedding for candidate in self.candidates]).astype(
            np.float32,
            copy=False,
        )
        similarities = matrix @ query
        self.candidates = [
            candidate
            for candidate, similarity in zip(self.candidates, similarities)
            if float(similarity) < self.similarity_threshold
        ]

    def observe(
        self, embedding: np.ndarray, now_monotonic: float
    ) -> tuple[np.ndarray | None, float]:
        query = normalize_embedding(embedding)
        if self.delay_sec <= 0.0:
            return query, 0.0

        self._prune_stale(now_monotonic)
        if not self.candidates:
            self.candidates.append(
                PendingCandidate(
                    embedding=query,
                    first_seen=now_monotonic,
                    last_seen=now_monotonic,
                )
            )
            return None, self.delay_sec

        matrix = np.vstack([candidate.embedding for candidate in self.candidates]).astype(
            np.float32,
            copy=False,
        )
        similarities = matrix @ query
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score < self.similarity_threshold:
            self.candidates.append(
                PendingCandidate(
                    embedding=query,
                    first_seen=now_monotonic,
                    last_seen=now_monotonic,
                )
            )
            return None, self.delay_sec

        candidate = self.candidates[best_idx]
        candidate.last_seen = now_monotonic
        # Candidate centroid smooths noisy frame-by-frame embeddings while waiting.
        candidate.embedding = normalize_embedding((0.9 * candidate.embedding) + (0.1 * query))

        elapsed = now_monotonic - candidate.first_seen
        remaining = max(0.0, self.delay_sec - elapsed)
        if elapsed >= self.delay_sec:
            final_embedding = candidate.embedding.copy()
            del self.candidates[best_idx]
            return final_embedding, 0.0
        return None, remaining


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
        self.match_threshold = match_threshold
        self.margin = margin
        self.commit_every_n = max(1, commit_every_n)
        self.commit_every_sec = max(0.1, commit_every_sec)
        self.centroid_alpha = float(np.clip(centroid_alpha, 0.0, 1.0))

        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        # WAL keeps reads and writes from blocking each other and improves write throughput.
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
                print(
                    f"Skipping person_id={person_id}: embedding dim {vec.size} != {expected_dim}"
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
            # executemany keeps the update path in C instead of Python loops for each row.
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

    def _append_person_cache(
        self, person_id: int, name: str, embedding: np.ndarray
    ) -> None:
        embedding = normalize_embedding(embedding)
        if self.embedding_matrix.size == 0:
            self.embedding_matrix = embedding.reshape(1, -1)
        else:
            if embedding.shape[0] != self.embedding_matrix.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: {embedding.shape[0]} != "
                    f"{self.embedding_matrix.shape[1]}"
                )
            self.embedding_matrix = np.vstack((self.embedding_matrix, embedding))

        self.person_ids = np.append(self.person_ids, np.int64(person_id))
        self.names.append(name)
        self.id_to_index[int(person_id)] = len(self.names) - 1

    def find_best_match(self, face_embedding: np.ndarray) -> MatchResult | None:
        if self.embedding_matrix.size == 0:
            return None

        query = normalize_embedding(face_embedding)
        # Embeddings are normalized, so dot product equals cosine similarity.
        # Matrix-vector multiply is vectorized NumPy and much faster than Python loops.
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

        # Margin check reduces false accepts when top candidates are too close.
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
        # Running centroid adapts to pose/lighting changes and lowers false rejects over time.
        updated = normalize_embedding(
            ((1.0 - self.centroid_alpha) * current) + (self.centroid_alpha * query)
        )
        self.embedding_matrix[idx] = updated

        self.pending_updates[person_id] = (updated, now_iso())
        self.pending_ops += 1
        self._flush_if_needed()


def is_valid_face_detection(
    face: Any,
    frame_shape: tuple[int, int, int],
    bbox_xyxy: np.ndarray | None = None,
) -> bool:
    if float(getattr(face, "det_score", 0.0)) < DETECTION_SCORE_THRESHOLD:
        return False

    if bbox_xyxy is None:
        x1, y1, x2, y2 = map(float, face.bbox)
    else:
        x1, y1, x2, y2 = map(float, bbox_xyxy)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < MIN_FACE_SIZE_PX or height < MIN_FACE_SIZE_PX:
        return False

    aspect_ratio = max(width / (height + 1e-8), height / (width + 1e-8))
    if aspect_ratio > MAX_FACE_ASPECT_RATIO:
        return False

    frame_h, frame_w = frame_shape[:2]
    area_ratio = (width * height) / float(frame_w * frame_h)
    if area_ratio < MIN_FACE_AREA_RATIO:
        return False

    # Extra boundary guard prevents partial out-of-frame boxes from polluting embeddings.
    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return False

    return True


def build_detection_frame(frame: np.ndarray, scale: float) -> tuple[np.ndarray, float, float]:
    if not (0.0 < scale <= 1.0):
        raise ValueError("DETECT_FRAME_SCALE must be in (0, 1].")
    if abs(scale - 1.0) < 1e-6:
        return frame, 1.0, 1.0

    frame_h, frame_w = frame.shape[:2]
    det_w = max(1, int(frame_w * scale))
    det_h = max(1, int(frame_h * scale))
    resized = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
    scale_x = frame_w / float(det_w)
    scale_y = frame_h / float(det_h)
    return resized, scale_x, scale_y


def scale_bbox_to_frame(bbox_xyxy: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    bbox = np.asarray(bbox_xyxy, dtype=np.float32).copy()
    bbox[0] *= scale_x
    bbox[2] *= scale_x
    bbox[1] *= scale_y
    bbox[3] *= scale_y
    return bbox


def draw_label(frame: np.ndarray, bbox_xyxy: np.ndarray, label: str) -> None:
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def process_faces_on_frame(
    app: FaceAnalysis,
    db_index: FaceDBIndex,
    new_person_gate: NewPersonGate,
    frame: np.ndarray,
) -> list[tuple[np.ndarray, str]]:
    detect_frame, scale_x, scale_y = build_detection_frame(frame, DETECT_FRAME_SCALE)
    faces = app.get(detect_frame)
    draw_items: list[tuple[np.ndarray, str]] = []
    now_monotonic = time.monotonic()
    new_person_gate.tick(now_monotonic)

    for face in faces:
        bbox = scale_bbox_to_frame(face.bbox, scale_x, scale_y)
        if not is_valid_face_detection(face, frame.shape, bbox_xyxy=bbox):
            continue

        embedding = normalize_embedding(face.normed_embedding.astype(np.float32, copy=False))
        match = db_index.find_best_match(embedding)
        if match is None:
            candidate_embedding, remaining_sec = new_person_gate.observe(
                embedding=embedding,
                now_monotonic=time.monotonic(),
            )
            if candidate_embedding is None:
                draw_items.append((bbox, f"unknown ({remaining_sec:.1f}s)"))
                continue
            new_person = db_index.insert_new_person(candidate_embedding)
            match = new_person
            print(
                f"New person saved after {NEW_PERSON_SAVE_DELAY_SEC:.1f}s hold "
                f"-> ID {new_person.person_id} / {new_person.name}"
            )
        else:
            new_person_gate.discard_if_similar(embedding=embedding, now_monotonic=time.monotonic())
            db_index.update_seen(match.person_id, embedding)

        label = f"{match.name} (ID:{match.person_id})"
        draw_items.append((bbox, label))

    return draw_items


def main() -> None:
    cv2.namedWindow("Face Recognition + DB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition + DB", DISPLAY_W, DISPLAY_H)

    use_cuda = torch.cuda.is_available()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    print(f"Using providers: {providers}")

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(
        ctx_id=0 if use_cuda else -1,
        det_size=DET_SIZE,
        det_thresh=DETECTION_SCORE_THRESHOLD,
    )

    db_index = FaceDBIndex(
        db_path=DB_PATH,
        match_threshold=MATCH_THRESHOLD,
        margin=MARGIN,
        commit_every_n=COMMIT_EVERY_N,
        commit_every_sec=COMMIT_EVERY_SEC,
        centroid_alpha=CENTROID_ALPHA,
    )
    new_person_gate = NewPersonGate(
        delay_sec=NEW_PERSON_SAVE_DELAY_SEC,
        similarity_threshold=PENDING_NEW_SIM_THRESHOLD,
        stale_sec=PENDING_NEW_STALE_SEC,
    )
    print(f"Loaded {db_index.person_ids.size} known people from {DB_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        db_index.close()
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_idx = 0
    last_draw_items: list[tuple[np.ndarray, str]] = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            should_detect = (frame_idx % max(1, DETECT_EVERY_N)) == 0
            if should_detect:
                last_draw_items = process_faces_on_frame(app, db_index, new_person_gate, frame)
            db_index.maybe_flush()

            for bbox, label in last_draw_items:
                draw_label(frame, bbox, label)

            display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Face Recognition + DB", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        db_index.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
