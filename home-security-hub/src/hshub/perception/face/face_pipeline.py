"""Face processing pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING
from typing import Any

import cv2
import numpy as np

from hshub.config import RuntimeConfig
from hshub.perception.face.facedb_sqlite import FaceDBIndex
from hshub.perception.face.facedb_sqlite import normalize_embedding
from hshub.perception.face.tuning_logger import TuningEventLogger
from hshub.types import DrawItem
from hshub.types import PendingCandidate

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis
else:
    FaceAnalysis = Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceFrameSummary:
    faces_seen: int
    faces_kept: int


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
    ) -> tuple[np.ndarray | None, float, float]:
        query = normalize_embedding(embedding)
        if self.delay_sec <= 0.0:
            return query, 0.0, 1.0

        self._prune_stale(now_monotonic)
        if not self.candidates:
            self.candidates.append(
                PendingCandidate(
                    embedding=query,
                    first_seen=now_monotonic,
                    last_seen=now_monotonic,
                )
            )
            return None, self.delay_sec, float("nan")

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
            return None, self.delay_sec, best_score

        candidate = self.candidates[best_idx]
        candidate.last_seen = now_monotonic
        candidate.embedding = normalize_embedding((0.9 * candidate.embedding) + (0.1 * query))

        elapsed = now_monotonic - candidate.first_seen
        remaining = max(0.0, self.delay_sec - elapsed)
        if elapsed >= self.delay_sec:
            final_embedding = candidate.embedding.copy()
            del self.candidates[best_idx]
            return final_embedding, 0.0, best_score
        return None, remaining, best_score


def normalized_gaussian_kernel(kernel_size: int) -> int:
    kernel = max(1, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    return kernel


def compute_face_blur_score(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    gaussian_kernel: int,
    gaussian_sigma: float,
) -> float:
    frame_h, frame_w = frame.shape[:2]
    x1 = int(np.floor(float(bbox_xyxy[0])))
    y1 = int(np.floor(float(bbox_xyxy[1])))
    x2 = int(np.ceil(float(bbox_xyxy[2])))
    y2 = int(np.ceil(float(bbox_xyxy[3])))

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w, x2))
    y2 = max(0, min(frame_h, y2))
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return float("nan")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    kernel = normalized_gaussian_kernel(gaussian_kernel)
    if kernel > 1 or gaussian_sigma > 0.0:
        gray = cv2.GaussianBlur(
            gray,
            (kernel, kernel),
            sigmaX=gaussian_sigma,
            sigmaY=gaussian_sigma,
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_face_detection(
    face: Any,
    frame: np.ndarray,
    config: RuntimeConfig,
    bbox_xyxy: np.ndarray | None = None,
) -> tuple[bool, str, float, float, float, float, float, float]:
    det_score = float(getattr(face, "det_score", 0.0))

    if bbox_xyxy is None:
        x1, y1, x2, y2 = map(float, face.bbox)
    else:
        x1, y1, x2, y2 = map(float, bbox_xyxy)

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    aspect_ratio = max(width / (height + 1e-8), height / (width + 1e-8))

    frame_h, frame_w = frame.shape[:2]
    area_ratio = (width * height) / float(frame_w * frame_h)
    in_frame = x1 >= 0 and y1 >= 0 and x2 <= frame_w and y2 <= frame_h
    blur_score = (
        compute_face_blur_score(
            frame=frame,
            bbox_xyxy=np.asarray([x1, y1, x2, y2], dtype=np.float32),
            gaussian_kernel=config.blur_gaussian_kernel,
            gaussian_sigma=config.blur_gaussian_sigma,
        )
        if in_frame
        else float("nan")
    )

    if det_score < config.detection_score_threshold:
        return (
            False,
            "det_score",
            det_score,
            width,
            height,
            aspect_ratio,
            area_ratio,
            blur_score,
        )
    if width < config.min_face_size_px or height < config.min_face_size_px:
        return (
            False,
            "min_size",
            det_score,
            width,
            height,
            aspect_ratio,
            area_ratio,
            blur_score,
        )
    if aspect_ratio > config.max_face_aspect_ratio:
        return (
            False,
            "aspect_ratio",
            det_score,
            width,
            height,
            aspect_ratio,
            area_ratio,
            blur_score,
        )
    if area_ratio < config.min_face_area_ratio:
        return (
            False,
            "area_ratio",
            det_score,
            width,
            height,
            aspect_ratio,
            area_ratio,
            blur_score,
        )
    if not in_frame:
        return (
            False,
            "out_of_frame",
            det_score,
            width,
            height,
            aspect_ratio,
            area_ratio,
            blur_score,
        )
    if config.min_blur_laplacian_var > 0.0:
        if not np.isfinite(blur_score):
            return (
                False,
                "blur_unavailable",
                det_score,
                width,
                height,
                aspect_ratio,
                area_ratio,
                blur_score,
            )
        if blur_score < config.min_blur_laplacian_var:
            return (
                False,
                "blur",
                det_score,
                width,
                height,
                aspect_ratio,
                area_ratio,
                blur_score,
            )

    return True, "accepted", det_score, width, height, aspect_ratio, area_ratio, blur_score


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
    frame_idx: int,
    tuning_logger: TuningEventLogger | None = None,
) -> tuple[list[DrawItem], FaceFrameSummary]:
    detect_frame, scale_x, scale_y = build_detection_frame(frame, config.detect_frame_scale)
    faces = app.get(detect_frame)
    draw_items: list[DrawItem] = []
    faces_seen = 0
    faces_kept = 0
    now_monotonic = time.monotonic()
    new_person_gate.tick(now_monotonic)

    for face in faces:
        faces_seen += 1
        bbox = scale_bbox_to_frame(face.bbox, scale_x, scale_y)
        (
            valid_detection,
            reject_reason,
            det_score,
            width,
            height,
            aspect_ratio,
            area_ratio,
            blur_score,
        ) = evaluate_face_detection(face=face, frame=frame, config=config, bbox_xyxy=bbox)
        if not valid_detection:
            if tuning_logger is not None:
                tuning_logger.log_face(
                    frame_idx=frame_idx,
                    det_score=det_score,
                    face_w=width,
                    face_h=height,
                    aspect_ratio=aspect_ratio,
                    area_ratio=area_ratio,
                    blur_score=blur_score,
                    face_valid=False,
                    reject_reason=reject_reason,
                    best_score=float("nan"),
                    second_score=float("nan"),
                    margin_gap=float("nan"),
                    match_pass=False,
                    matched_person_id=None,
                    is_new_person=False,
                    pending_similarity=float("nan"),
                    pending_remaining_sec=0.0,
                    embedding_update_l2=float("nan"),
                )
            continue

        embedding = normalize_embedding(face.normed_embedding.astype(np.float32, copy=False))
        match, best_score, second_score = db_index.find_best_match_with_scores(embedding)
        margin_gap = (
            best_score - second_score
            if np.isfinite(best_score) and np.isfinite(second_score) and second_score >= 0.0
            else float("nan")
        )
        margin_ok = not np.isfinite(margin_gap) or margin_gap >= config.margin
        match_pass = bool(np.isfinite(best_score) and best_score >= config.match_threshold and margin_ok)

        pending_similarity = float("nan")
        pending_remaining_sec = 0.0
        matched_person_id: int | None = None
        is_new_person = False
        embedding_update_l2 = float("nan")
        label: str
        if match is None:
            candidate_embedding, remaining_sec, pending_similarity = new_person_gate.observe(
                embedding=embedding,
                now_monotonic=time.monotonic(),
            )
            pending_remaining_sec = remaining_sec
            if candidate_embedding is None:
                label = f"unknown ({remaining_sec:.1f}s)"
            else:
                new_person = db_index.insert_new_person(candidate_embedding)
                match = new_person
                is_new_person = True
                LOGGER.info(
                    "New person saved after %.1fs hold -> ID %s / %s",
                    config.new_person_save_delay_sec,
                    new_person.person_id,
                    new_person.name,
                )
        else:
            new_person_gate.discard_if_similar(embedding=embedding, now_monotonic=time.monotonic())
            embedding_update_l2 = db_index.update_seen(match.person_id, embedding)

        if match is not None:
            matched_person_id = match.person_id
            label = f"{match.name} (ID:{match.person_id})"
        draw_items.append((bbox, label))
        faces_kept += 1

        if tuning_logger is not None:
            tuning_logger.log_face(
                frame_idx=frame_idx,
                det_score=det_score,
                face_w=width,
                face_h=height,
                aspect_ratio=aspect_ratio,
                area_ratio=area_ratio,
                blur_score=blur_score,
                face_valid=True,
                reject_reason="accepted",
                best_score=best_score,
                second_score=second_score,
                margin_gap=margin_gap,
                match_pass=match_pass,
                matched_person_id=matched_person_id,
                is_new_person=is_new_person,
                pending_similarity=pending_similarity,
                pending_remaining_sec=pending_remaining_sec,
                embedding_update_l2=embedding_update_l2,
            )

    return draw_items, FaceFrameSummary(faces_seen=faces_seen, faces_kept=faces_kept)
