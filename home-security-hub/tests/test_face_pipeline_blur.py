from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np

from hshub.config import DEFAULT_CONFIG
from hshub.perception.face.face_pipeline import compute_face_blur_score
from hshub.perception.face.face_pipeline import evaluate_face_detection
from hshub.perception.face.face_pipeline import normalized_gaussian_kernel


def make_checkerboard(size: int = 128, block: int = 4) -> np.ndarray:
    yy, xx = np.indices((size, size))
    board = (((xx // block) + (yy // block)) % 2).astype(np.uint8) * 255
    return np.stack((board, board, board), axis=-1)


def test_normalized_gaussian_kernel_forces_odd() -> None:
    assert normalized_gaussian_kernel(4) == 5
    assert normalized_gaussian_kernel(5) == 5


def test_blur_score_drops_after_gaussian_blur() -> None:
    image = make_checkerboard()
    blurred = cv2.GaussianBlur(image, (11, 11), sigmaX=3.0)
    bbox = np.asarray([16, 16, 112, 112], dtype=np.float32)

    sharp_score = compute_face_blur_score(
        frame=image,
        bbox_xyxy=bbox,
        gaussian_kernel=5,
        gaussian_sigma=0.0,
    )
    blurred_score = compute_face_blur_score(
        frame=blurred,
        bbox_xyxy=bbox,
        gaussian_kernel=5,
        gaussian_sigma=0.0,
    )
    assert sharp_score > blurred_score


def test_evaluate_face_detection_rejects_blurry_face() -> None:
    image = make_checkerboard()
    blurred = cv2.GaussianBlur(image, (11, 11), sigmaX=3.0)
    bbox = np.asarray([16, 16, 112, 112], dtype=np.float32)
    face = SimpleNamespace(det_score=0.95, bbox=bbox)

    sharp_score = compute_face_blur_score(
        frame=image,
        bbox_xyxy=bbox,
        gaussian_kernel=5,
        gaussian_sigma=0.0,
    )
    blurred_score = compute_face_blur_score(
        frame=blurred,
        bbox_xyxy=bbox,
        gaussian_kernel=5,
        gaussian_sigma=0.0,
    )
    threshold = float((sharp_score + blurred_score) * 0.5)
    config = replace(
        DEFAULT_CONFIG,
        detection_score_threshold=0.5,
        min_face_size_px=10,
        min_face_area_ratio=0.0,
        min_blur_laplacian_var=threshold,
        blur_gaussian_kernel=5,
        blur_gaussian_sigma=0.0,
    )

    sharp_valid, sharp_reason, *_ = evaluate_face_detection(
        face=face,
        frame=image,
        config=config,
        bbox_xyxy=bbox,
    )
    blurred_valid, blurred_reason, *_ = evaluate_face_detection(
        face=face,
        frame=blurred,
        config=config,
        bbox_xyxy=bbox,
    )

    assert sharp_valid
    assert sharp_reason == "accepted"
    assert not blurred_valid
    assert blurred_reason == "blur"
