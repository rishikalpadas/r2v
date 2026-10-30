"""Stage parameter dataclasses and the preset table."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Literal

from engine.errors import InputError

LAYER_MODES = ("stacked", "cutout")


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


@dataclass(frozen=True)
class FitParams:
    enabled: bool = True
    tolerance_px: float = 0.75
    complexity_weight: float = 2.0
    axis_snap_deg: float = 3.0
    smooth_sigma: float = 1.0
    corner_angle_deg: float = 35.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        numeric = (self.tolerance_px, self.complexity_weight, self.axis_snap_deg,
                   self.smooth_sigma, self.corner_angle_deg)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in numeric):
            raise ValueError("fit parameters must be numbers")
        if not self.tolerance_px > 0:
            raise ValueError("tolerance_px must be > 0")
        if not self.complexity_weight >= 0:
            raise ValueError("complexity_weight must be >= 0")
        if not 0 <= self.axis_snap_deg < 45:
            raise ValueError("axis_snap_deg must be in [0, 45)")
        if not self.smooth_sigma >= 0:
            raise ValueError("smooth_sigma must be >= 0")
        if not 0 < self.corner_angle_deg < 180:
            raise ValueError("corner_angle_deg must be in (0, 180)")


@dataclass(frozen=True)
class ExportParams:
    layer_mode: Literal["stacked", "cutout"] = "stacked"
    precision: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.precision, int) or isinstance(self.precision, bool):
            raise ValueError("precision must be an integer")
        if self.layer_mode not in LAYER_MODES:
            raise ValueError(f"layer_mode must be one of {LAYER_MODES}")
        if not 0 <= self.precision <= 8:
            raise ValueError("precision must be between 0 and 8")


@dataclass(frozen=True)
class Preset:
    name: str
    stages: tuple[str, ...]
    segment: SegmentParams
    fit: FitParams
    export: ExportParams


PRESETS: dict[str, Preset] = {
    "logo": Preset(
        name="logo",
        stages=("segment", "trace", "fit", "export"),
        segment=SegmentParams(),
        fit=FitParams(),
        export=ExportParams(),
    ),
}

PRESET_ALIASES: Mapping[str, str] = {"auto": "logo"}
_OVERRIDE_GROUPS = ("segment", "fit", "export")


def resolve_preset(name: str) -> Preset:
    key = PRESET_ALIASES.get(name, name)
    try:
        return PRESETS[key]
    except KeyError:
        raise InputError(f"unknown preset {name!r}") from None


def apply_overrides(preset: Preset, overrides: Mapping[str, object]) -> Preset:
    """Return a new preset with `"<group>.<field>"` overrides applied; `preset` is untouched."""
    updated = {group: getattr(preset, group) for group in _OVERRIDE_GROUPS}
    for key, value in overrides.items():
        group, _, field = key.partition(".")
        if group not in updated or field not in {f.name for f in fields(updated[group])}:
            raise InputError(f"unknown override {key!r}")
        try:
            updated[group] = replace(updated[group], **{field: value})
        except (ValueError, TypeError) as exc:
            raise InputError(f"invalid value for {key!r}: {exc}") from exc
    return replace(preset, **updated)
