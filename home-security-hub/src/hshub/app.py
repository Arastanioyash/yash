from __future__ import annotations

# Support running as `python src/hshub/app.py` from the repository root.
if __package__ in (None, ""):
    import sys

    script_dir = __file__.replace("\\", "/").rsplit("/", 1)[0].rstrip("/")
    src_dir = script_dir.rsplit("/", 1)[0]
    if sys.path and sys.path[0].replace("\\", "/").rstrip("/") == script_dir:
        # Remove `src/hshub` to avoid shadowing stdlib `types` with `hshub/types.py`.
        sys.path.pop(0)
    normalized_paths = [path.replace("\\", "/").rstrip("/") for path in sys.path]
    if src_dir.rstrip("/") not in normalized_paths:
        sys.path.insert(0, src_dir)

import logging
import os
from dataclasses import replace
import time

import cv2
import torch
from insightface.app import FaceAnalysis

from hshub.config import DEFAULT_CONFIG
from hshub.config import RuntimeConfig
from hshub.io.video_source import VideoSourceError
from hshub.io.video_source import open_video_capture
from hshub.logging_utils import configure_logging
from hshub.perception.face.face_pipeline import NewPersonGate
from hshub.perception.face.face_pipeline import process_faces_on_frame
from hshub.perception.face.facedb_sqlite import FaceDBIndex
from hshub.perception.face.tuning_logger import TuningEventLogger
from hshub.types import DrawItem
from hshub.ui.overlay import draw_items

LOGGER = logging.getLogger(__name__)


def _parse_bool(raw_value: str) -> bool:
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw_value!r}")


def _parse_det_size(raw_value: str) -> tuple[int, int]:
    normalized = raw_value.strip().lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("det_size must be formatted as WIDTH,HEIGHT or WIDTHxHEIGHT.")
    return int(parts[0]), int(parts[1])


def _parse_optional_path(raw_value: str) -> str | None:
    stripped = raw_value.strip()
    if not stripped or stripped.lower() == "none":
        return None
    return stripped


def apply_env_overrides(config: RuntimeConfig) -> RuntimeConfig:
    parser_by_env: dict[str, tuple[str, object]] = {
        "HSHUB_VIDEO_SOURCE": ("video_path", str),
        "HSHUB_FALLBACK_TO_WEBCAM": ("fallback_to_webcam", _parse_bool),
        "HSHUB_FALLBACK_WEBCAM_INDEX": ("fallback_webcam_index", int),
        "HSHUB_MATCH_THRESHOLD": ("match_threshold", float),
        "HSHUB_MARGIN": ("margin", float),
        "HSHUB_COMMIT_EVERY_N": ("commit_every_n", int),
        "HSHUB_COMMIT_EVERY_SEC": ("commit_every_sec", float),
        "HSHUB_DETECT_EVERY_N": ("detect_every_n", int),
        "HSHUB_DETECT_FRAME_SCALE": ("detect_frame_scale", float),
        "HSHUB_DET_SIZE": ("det_size", _parse_det_size),
        "HSHUB_DETECTION_SCORE_THRESHOLD": ("detection_score_threshold", float),
        "HSHUB_MIN_FACE_SIZE_PX": ("min_face_size_px", int),
        "HSHUB_MAX_FACE_ASPECT_RATIO": ("max_face_aspect_ratio", float),
        "HSHUB_MIN_FACE_AREA_RATIO": ("min_face_area_ratio", float),
        "HSHUB_CENTROID_ALPHA": ("centroid_alpha", float),
        "HSHUB_NEW_PERSON_SAVE_DELAY_SEC": ("new_person_save_delay_sec", float),
        "HSHUB_PENDING_NEW_SIM_THRESHOLD": ("pending_new_sim_threshold", float),
        "HSHUB_PENDING_NEW_STALE_SEC": ("pending_new_stale_sec", float),
        "HSHUB_MIN_BLUR_LAPLACIAN_VAR": ("min_blur_laplacian_var", float),
        "HSHUB_BLUR_GAUSSIAN_KERNEL": ("blur_gaussian_kernel", int),
        "HSHUB_BLUR_GAUSSIAN_SIGMA": ("blur_gaussian_sigma", float),
        "HSHUB_TUNING_LOG_PATH": ("tuning_log_path", _parse_optional_path),
    }
    overrides: dict[str, object] = {}
    for env_name, (field_name, parser) in parser_by_env.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        parsed_value = parser(raw_value) if callable(parser) else parser(raw_value)
        overrides[field_name] = parsed_value

    if not overrides:
        return config
    return replace(config, **overrides)


def run(config: RuntimeConfig) -> None:
    config.validate()
    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.window_name, config.display_width, config.display_height)

    use_cuda = torch.cuda.is_available()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    LOGGER.info("Using providers: %s", providers)

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(
        ctx_id=0 if use_cuda else -1,
        det_size=config.det_size,
        det_thresh=config.detection_score_threshold,
    )
    tuning_logger = TuningEventLogger(config=config)

    db_index = FaceDBIndex(
        db_path=config.db_path,
        match_threshold=config.match_threshold,
        margin=config.margin,
        commit_every_n=config.commit_every_n,
        commit_every_sec=config.commit_every_sec,
        centroid_alpha=config.centroid_alpha,
        flush_event_hook=tuning_logger.log_flush if tuning_logger.enabled else None,
    )
    new_person_gate = NewPersonGate(
        delay_sec=config.new_person_save_delay_sec,
        similarity_threshold=config.pending_new_sim_threshold,
        stale_sec=config.pending_new_stale_sec,
    )
    LOGGER.info("Loaded %s known people from %s", db_index.person_ids.size, config.db_path)

    fallback_webcam_index = config.fallback_webcam_index if config.fallback_to_webcam else None
    try:
        cap, source_label = open_video_capture(
            video_source=config.video_path,
            fallback_webcam_index=fallback_webcam_index,
        )
    except VideoSourceError as exc:
        db_index.close()
        tuning_logger.close()
        raise RuntimeError(str(exc)) from exc
    LOGGER.info("Opened video source: %s", source_label)

    frame_idx = 0
    last_draw_items: list[DrawItem] = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            should_detect = (frame_idx % config.detect_every_n) == 0
            detect_ms = float("nan")
            faces_seen = 0
            faces_kept = 0
            if should_detect:
                started = time.perf_counter()
                last_draw_items, frame_summary = process_faces_on_frame(
                    app=app,
                    db_index=db_index,
                    new_person_gate=new_person_gate,
                    frame=frame,
                    config=config,
                    frame_idx=frame_idx,
                    tuning_logger=tuning_logger,
                )
                detect_ms = (time.perf_counter() - started) * 1000.0
                faces_seen = frame_summary.faces_seen
                faces_kept = frame_summary.faces_kept
            db_index.maybe_flush()
            tuning_logger.log_frame(
                frame_idx=frame_idx,
                detect_ran=should_detect,
                detect_ms=detect_ms,
                faces_seen=faces_seen,
                faces_kept=faces_kept,
            )

            draw_items(frame, last_draw_items)
            display_frame = cv2.resize(
                frame,
                (config.display_width, config.display_height),
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imshow(config.window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        db_index.close()
        tuning_logger.close()
        cv2.destroyAllWindows()


def main() -> None:
    configure_logging()
    config = apply_env_overrides(DEFAULT_CONFIG)
    run(config)


if __name__ == "__main__":
    main()
