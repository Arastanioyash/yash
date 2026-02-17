"""Shared dataclasses for detections/events/decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    person_id: int
    name: str
    best_score: float
    second_score: float


@dataclass
class PendingCandidate:
    embedding: np.ndarray
    first_seen: float
    last_seen: float


DrawItem: TypeAlias = tuple[np.ndarray, str]
