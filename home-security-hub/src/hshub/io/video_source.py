"""Video source helpers for file, stream, and webcam inputs."""

from __future__ import annotations

from pathlib import Path

import cv2


class VideoSourceError(RuntimeError):
    """Raised when no configured video source can be opened."""


def _repo_root() -> Path:
    # .../home-security-hub/src/hshub/io/video_source.py -> home-security-hub
    return Path(__file__).resolve().parents[3]


def _is_stream_uri(source: str) -> bool:
    normalized = source.lower()
    return normalized.startswith(("rtsp://", "http://", "https://", "rtmp://"))


def _build_candidates(video_source: str) -> list[tuple[int | str, str]]:
    source = video_source.strip()
    if not source:
        return []

    if source.isdigit():
        index = int(source)
        return [(index, f"webcam index {index}")]

    lowered = source.lower()
    if lowered in {"webcam", "camera", "cam"}:
        return [(0, "webcam index 0")]

    if _is_stream_uri(source):
        return [(source, source)]

    source_path = Path(source)
    candidates: list[tuple[int | str, str]] = []
    seen: set[str] = set()
    path_candidates = (
        [source_path]
        if source_path.is_absolute()
        else [Path.cwd() / source_path, _repo_root() / source_path]
    )
    for candidate in path_candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append((resolved, resolved))
    return candidates


def open_video_capture(
    video_source: str,
    fallback_webcam_index: int | None = None,
) -> tuple[cv2.VideoCapture, str]:
    """Open source by trying file/URI/webcam candidates in order."""
    candidates = _build_candidates(video_source)
    attempted_labels = [label for _, label in candidates]

    for candidate, label in candidates:
        cap = cv2.VideoCapture(candidate)
        if cap.isOpened():
            return cap, label
        cap.release()

    if fallback_webcam_index is not None:
        fallback_label = f"webcam index {fallback_webcam_index}"
        attempted_labels.append(fallback_label)
        cap = cv2.VideoCapture(fallback_webcam_index)
        if cap.isOpened():
            return cap, fallback_label
        cap.release()

    attempted = ", ".join(attempted_labels) if attempted_labels else "<none>"
    raise VideoSourceError(
        "Could not open configured video source "
        f"'{video_source}'. Attempted: {attempted}. "
        "Set `video_path` to a valid file, RTSP URL, or webcam index (e.g. '0')."
    )
