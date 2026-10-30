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

    def __post_init__(self) -> None:
        if self.max_colors < 1:
            raise ValueError("max_colors must be >= 1")
        if self.min_region_px < 0:
            raise ValueError("min_region_px must be >= 0")
        if self.merge_delta_e < 0:
            raise ValueError("merge_delta_e must be >= 0")
        if self.sample_px < 1:
            raise ValueError("sample_px must be >= 1")
