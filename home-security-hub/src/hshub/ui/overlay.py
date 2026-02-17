"""UI overlay rendering helpers."""

from __future__ import annotations

import cv2
import numpy as np

from hshub.types import DrawItem


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


def draw_items(frame: np.ndarray, items: list[DrawItem]) -> None:
    for bbox, label in items:
        draw_label(frame, bbox, label)
