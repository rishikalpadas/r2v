"""Stage parameter dataclasses (preset table is added in a later task)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentParams:
    max_colors: int = 16
    merge_delta_e: float = 8.0
    min_region_px: int = 4
    sample_px: int = 200_000
    seed: int = 0
