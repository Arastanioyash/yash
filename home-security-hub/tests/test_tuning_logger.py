from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from hshub.config import DEFAULT_CONFIG
from hshub.perception.face.tuning_logger import TuningEventLogger


def test_tuning_logger_writes_face_frame_and_flush_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "tuning_events.csv"
    config = replace(DEFAULT_CONFIG, tuning_log_path=str(log_path))
    logger = TuningEventLogger(config)

    logger.log_frame(
        frame_idx=1,
        detect_ran=True,
        detect_ms=12.5,
        faces_seen=2,
        faces_kept=1,
    )
    logger.log_face(
        frame_idx=1,
        det_score=0.9,
        face_w=64.0,
        face_h=64.0,
        aspect_ratio=1.0,
        area_ratio=0.01,
        blur_score=140.0,
        face_valid=True,
        reject_reason="accepted",
        best_score=0.8,
        second_score=0.6,
        margin_gap=0.2,
        match_pass=True,
        matched_person_id=3,
        is_new_person=False,
        pending_similarity=0.7,
        pending_remaining_sec=0.0,
        embedding_update_l2=0.03,
    )
    logger.log_flush(
        reason="ops",
        pending_ops_before_flush=50,
        elapsed_since_commit_sec=0.8,
        force_flush=False,
    )
    logger.close()

    assert log_path.exists()
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert {row["event_type"] for row in rows} == {"frame", "face", "flush"}
