"""Face processing pipeline orchestration."""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from hshub.config import RuntimeConfig
from hshub.perception.face.facedb_sqlite import FaceDBIndex
from hshub.perception.face.facedb_sqlite import normalize_embedding
from hshub.types import DrawItem
from hshub.types import PendingCandidate

LOGGER = logging.getLogger(__name__)


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
        candidate.embedding = normalize_embedding((0.9 * candidate.embedding) + (0.1 * query))

        elapsed = now_monotonic - candidate.first_seen
        remaining = max(0.0, self.delay_sec - elapsed)
        if elapsed >= self.delay_sec:
            final_embedding = candidate.embedding.copy()
            del self.candidates[best_idx]
            return final_embedding, 0.0
        return None, remaining


def is_valid_face_detection(
    face: Any,
    frame_shape: tuple[int, int, int],
    config: RuntimeConfig,
    bbox_xyxy: np.ndarray | None = None,
) -> bool:
    if float(getattr(face, "det_score", 0.0)) < config.detection_score_threshold:
        return False

    if bbox_xyxy is None:
        x1, y1, x2, y2 = map(float, face.bbox)
    else:
        x1, y1, x2, y2 = map(float, bbox_xyxy)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width < config.min_face_size_px or height < config.min_face_size_px:
        return False

    aspect_ratio = max(width / (height + 1e-8), height / (width + 1e-8))
    if aspect_ratio > config.max_face_aspect_ratio:
        return False

    frame_h, frame_w = frame_shape[:2]
    area_ratio = (width * height) / float(frame_w * frame_h)
    if area_ratio < config.min_face_area_ratio:
        return False

    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return False

    return True


def build_detection_frame(frame: np.ndarray, scale: float) -> tuple[np.ndarray, float, float]:
    if not (0.0 < scale <= 1.0):
        raise ValueError("detect_frame_scale must be in (0, 1].")
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


def process_faces_on_frame(
    app: FaceAnalysis,
    db_index: FaceDBIndex,
    new_person_gate: NewPersonGate,
    frame: np.ndarray,
    config: RuntimeConfig,
) -> list[DrawItem]:
    detect_frame, scale_x, scale_y = build_detection_frame(frame, config.detect_frame_scale)
    faces = app.get(detect_frame)
    draw_items: list[DrawItem] = []
    now_monotonic = time.monotonic()
    new_person_gate.tick(now_monotonic)

    for face in faces:
        bbox = scale_bbox_to_frame(face.bbox, scale_x, scale_y)
        if not is_valid_face_detection(face, frame.shape, config, bbox_xyxy=bbox):
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
            LOGGER.info(
                "New person saved after %.1fs hold -> ID %s / %s",
                config.new_person_save_delay_sec,
                new_person.person_id,
                new_person.name,
            )
        else:
            new_person_gate.discard_if_similar(embedding=embedding, now_monotonic=time.monotonic())
            db_index.update_seen(match.person_id, embedding)

        label = f"{match.name} (ID:{match.person_id})"
        draw_items.append((bbox, label))

    return draw_items
