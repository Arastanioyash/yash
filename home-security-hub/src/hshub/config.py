"""Configuration loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    db_path: str = "faces.db"
    video_path: str = "test1.mp4"
    fallback_to_webcam: bool = True
    fallback_webcam_index: int = 0
    match_threshold: float = 0.50
    margin: float = 0.04
    commit_every_n: int = 50
    commit_every_sec: float = 1.0
    detect_every_n: int = 1
    detect_frame_scale: float = 1.0
    det_size: tuple[int, int] = (640, 640)
    detection_score_threshold: float = 0.60
    min_face_size_px: int = 30
    max_face_aspect_ratio: float = 1.6
    min_face_area_ratio: float = 0.002
    centroid_alpha: float = 0.05
    new_person_save_delay_sec: float = 2.0
    pending_new_sim_threshold: float = 0.60
    pending_new_stale_sec: float = 2.0
    display_width: int = 1280
    display_height: int = 720
    window_name: str = "Face Recognition + DB"

    def validate(self) -> None:
        if not (0.0 < self.match_threshold <= 1.0):
            raise ValueError("match_threshold must be in (0, 1].")
        if self.margin < 0.0:
            raise ValueError("margin must be >= 0.")
        if self.commit_every_n < 1:
            raise ValueError("commit_every_n must be >= 1.")
        if self.commit_every_sec <= 0.0:
            raise ValueError("commit_every_sec must be > 0.")
        if self.detect_every_n < 1:
            raise ValueError("detect_every_n must be >= 1.")
        if not (0.0 < self.detect_frame_scale <= 1.0):
            raise ValueError("detect_frame_scale must be in (0, 1].")
        if self.det_size[0] <= 0 or self.det_size[1] <= 0:
            raise ValueError("det_size values must be > 0.")
        if not (0.0 <= self.detection_score_threshold <= 1.0):
            raise ValueError("detection_score_threshold must be in [0, 1].")
        if self.min_face_size_px < 1:
            raise ValueError("min_face_size_px must be >= 1.")
        if self.max_face_aspect_ratio <= 1.0:
            raise ValueError("max_face_aspect_ratio must be > 1.")
        if self.min_face_area_ratio < 0.0:
            raise ValueError("min_face_area_ratio must be >= 0.")
        if not (0.0 <= self.centroid_alpha <= 1.0):
            raise ValueError("centroid_alpha must be in [0, 1].")
        if self.new_person_save_delay_sec < 0.0:
            raise ValueError("new_person_save_delay_sec must be >= 0.")
        if not (0.0 <= self.pending_new_sim_threshold <= 1.0):
            raise ValueError("pending_new_sim_threshold must be in [0, 1].")
        if self.pending_new_stale_sec <= 0.0:
            raise ValueError("pending_new_stale_sec must be > 0.")
        if self.display_width < 1 or self.display_height < 1:
            raise ValueError("display dimensions must be >= 1.")
        if self.fallback_webcam_index < 0:
            raise ValueError("fallback_webcam_index must be >= 0.")


DEFAULT_CONFIG = RuntimeConfig()
