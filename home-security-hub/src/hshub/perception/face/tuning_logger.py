"""CSV logger for face-pipeline tuning and graph generation."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from hshub.config import RuntimeConfig


class TuningEventLogger:
    FIELDNAMES = [
        "event_ts_utc",
        "event_type",
        "frame_idx",
        "detect_ran",
        "detect_ms",
        "faces_seen",
        "faces_kept",
        "flush_reason",
        "pending_ops_before_flush",
        "elapsed_since_commit_sec",
        "force_flush",
        "det_score",
        "face_w",
        "face_h",
        "aspect_ratio",
        "area_ratio",
        "blur_score",
        "face_valid",
        "reject_reason",
        "best_score",
        "second_score",
        "margin_gap",
        "match_pass",
        "matched_person_id",
        "is_new_person",
        "pending_similarity",
        "pending_remaining_sec",
        "embedding_update_l2",
        "config_match_threshold",
        "config_margin",
        "config_commit_every_n",
        "config_commit_every_sec",
        "config_detect_every_n",
        "config_detect_frame_scale",
        "config_det_size_w",
        "config_det_size_h",
        "config_detection_score_threshold",
        "config_min_face_size_px",
        "config_max_face_aspect_ratio",
        "config_min_face_area_ratio",
        "config_centroid_alpha",
        "config_new_person_save_delay_sec",
        "config_pending_new_sim_threshold",
        "config_pending_new_stale_sec",
        "config_min_blur_laplacian_var",
        "config_blur_gaussian_kernel",
        "config_blur_gaussian_sigma",
    ]

    def __init__(self, config: RuntimeConfig) -> None:
        self.enabled = bool(config.tuning_log_path)
        self._rows_since_flush = 0
        self._file = None
        self._writer: csv.DictWriter[str] | None = None

        self._config_payload = {
            "config_match_threshold": config.match_threshold,
            "config_margin": config.margin,
            "config_commit_every_n": config.commit_every_n,
            "config_commit_every_sec": config.commit_every_sec,
            "config_detect_every_n": config.detect_every_n,
            "config_detect_frame_scale": config.detect_frame_scale,
            "config_det_size_w": config.det_size[0],
            "config_det_size_h": config.det_size[1],
            "config_detection_score_threshold": config.detection_score_threshold,
            "config_min_face_size_px": config.min_face_size_px,
            "config_max_face_aspect_ratio": config.max_face_aspect_ratio,
            "config_min_face_area_ratio": config.min_face_area_ratio,
            "config_centroid_alpha": config.centroid_alpha,
            "config_new_person_save_delay_sec": config.new_person_save_delay_sec,
            "config_pending_new_sim_threshold": config.pending_new_sim_threshold,
            "config_pending_new_stale_sec": config.pending_new_stale_sec,
            "config_min_blur_laplacian_var": config.min_blur_laplacian_var,
            "config_blur_gaussian_kernel": config.blur_gaussian_kernel,
            "config_blur_gaussian_sigma": config.blur_gaussian_sigma,
        }
        if not self.enabled:
            return

        assert config.tuning_log_path is not None
        log_path = Path(config.tuning_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = log_path.exists() and log_path.stat().st_size > 0
        self._file = log_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    @staticmethod
    def _format_value(value: Any) -> str | int | float:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return ""
            return value
        return value

    def _write_row(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._writer is None:
            return
        row = {key: "" for key in self.FIELDNAMES}
        row.update(self._config_payload)
        row["event_ts_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        for key, value in payload.items():
            if key in row:
                row[key] = self._format_value(value)
        self._writer.writerow(row)
        self._rows_since_flush += 1
        if self._file is not None and self._rows_since_flush >= 50:
            self._file.flush()
            self._rows_since_flush = 0

    def log_frame(
        self,
        frame_idx: int,
        detect_ran: bool,
        detect_ms: float,
        faces_seen: int,
        faces_kept: int,
    ) -> None:
        self._write_row(
            {
                "event_type": "frame",
                "frame_idx": frame_idx,
                "detect_ran": detect_ran,
                "detect_ms": detect_ms,
                "faces_seen": faces_seen,
                "faces_kept": faces_kept,
            }
        )

    def log_face(
        self,
        frame_idx: int,
        det_score: float,
        face_w: float,
        face_h: float,
        aspect_ratio: float,
        area_ratio: float,
        blur_score: float,
        face_valid: bool,
        reject_reason: str,
        best_score: float,
        second_score: float,
        margin_gap: float,
        match_pass: bool,
        matched_person_id: int | None,
        is_new_person: bool,
        pending_similarity: float,
        pending_remaining_sec: float,
        embedding_update_l2: float,
    ) -> None:
        self._write_row(
            {
                "event_type": "face",
                "frame_idx": frame_idx,
                "det_score": det_score,
                "face_w": face_w,
                "face_h": face_h,
                "aspect_ratio": aspect_ratio,
                "area_ratio": area_ratio,
                "blur_score": blur_score,
                "face_valid": face_valid,
                "reject_reason": reject_reason,
                "best_score": best_score,
                "second_score": second_score,
                "margin_gap": margin_gap,
                "match_pass": match_pass,
                "matched_person_id": matched_person_id,
                "is_new_person": is_new_person,
                "pending_similarity": pending_similarity,
                "pending_remaining_sec": pending_remaining_sec,
                "embedding_update_l2": embedding_update_l2,
            }
        )

    def log_flush(
        self,
        reason: str,
        pending_ops_before_flush: int,
        elapsed_since_commit_sec: float,
        force_flush: bool,
    ) -> None:
        self._write_row(
            {
                "event_type": "flush",
                "flush_reason": reason,
                "pending_ops_before_flush": pending_ops_before_flush,
                "elapsed_since_commit_sec": elapsed_since_commit_sec,
                "force_flush": force_flush,
            }
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
