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
from hshub.types import DrawItem
from hshub.ui.overlay import draw_items

LOGGER = logging.getLogger(__name__)


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

    db_index = FaceDBIndex(
        db_path=config.db_path,
        match_threshold=config.match_threshold,
        margin=config.margin,
        commit_every_n=config.commit_every_n,
        commit_every_sec=config.commit_every_sec,
        centroid_alpha=config.centroid_alpha,
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
            if should_detect:
                last_draw_items = process_faces_on_frame(
                    app=app,
                    db_index=db_index,
                    new_person_gate=new_person_gate,
                    frame=frame,
                    config=config,
                )
            db_index.maybe_flush()

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
        cv2.destroyAllWindows()


def main() -> None:
    configure_logging()
    config = DEFAULT_CONFIG
    env_video_source = os.getenv("HSHUB_VIDEO_SOURCE")
    if env_video_source:
        config = replace(config, video_path=env_video_source)
    run(config)


if __name__ == "__main__":
    main()
