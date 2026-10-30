"""Core immutable data types shared by all engine stages."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

Point = tuple[float, float]

TRANSPARENT = -1
OUTSIDE = -2


@dataclass(frozen=True)
class RasterImage:
    pixels: np.ndarray  # HxWx4 uint8
    original_size: tuple[int, int]  # (w, h)
    scale: float = 1.0
    color_space: str = "sRGB"


@dataclass(frozen=True)
class ImageProfile:
    kind: Literal["logo", "line_art", "photo", "mixed"]
    color_count: int
    noise_level: float
    preset: str


@dataclass(frozen=True)
class RegionMap:
    labels: np.ndarray  # HxW int32
    palette_rgb: np.ndarray  # Kx3 uint8
    palette_lab: np.ndarray  # Kx3 float64
    adjacency: frozenset[tuple[int, int]]
    gradient_regions: frozenset[int] = frozenset()


@dataclass(frozen=True)
class BoundaryCurve:
    """Polyline on the pixel-corner grid separating exactly two labels.

    Walking `points` in order, `left` is on the left and `right` on the right, where "left" of
    direction (dx, dy) is (-dy, dx) with coordinates taken as-is (x right, y down). Closed curves
    repeat their first point as their last.
    """

    id: int
    points: tuple[Point, ...]
    left: int
    right: int
    closed: bool


LoopRef = tuple[tuple[int, bool], ...]  # (curve_id, reversed) in walking order


@dataclass(frozen=True)
class BoundaryGraph:
    """Shared boundary curves plus, per region, its loops (loops[0] outer, rest holes)."""

    curves: tuple[BoundaryCurve, ...]
    region_loops: Mapping[int, tuple[LoopRef, ...]]


@dataclass(frozen=True)
class Line:
    p0: Point
    p1: Point


@dataclass(frozen=True)
class Arc:
    p0: Point
    p1: Point
    center: Point
    radius: float
    large_arc: bool
    sweep: bool


@dataclass(frozen=True)
class CubicBezier:
    p0: Point
    c1: Point
    c2: Point
    p1: Point


Segment = Line | Arc | CubicBezier


@dataclass(frozen=True)
class SolidFill:
    rgb: tuple[int, int, int]


Fill = SolidFill


@dataclass(frozen=True)
class Path:
    contours: tuple[tuple[Segment, ...], ...]  # contours[0] outer, rest holes
    fill: Fill
    z_order: int


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float
    angle_deg: float


@dataclass(frozen=True)
class Polygon:
    points: tuple[Point, ...]


PrimitiveShape = Circle | Ellipse | Rect | Polygon


@dataclass(frozen=True)
class Primitive:
    shape: PrimitiveShape
    fill: Fill
    z_order: int


@dataclass(frozen=True)
class FittedCurve:
    """A boundary curve replaced by fitted segments (or recognised as a primitive shape)."""

    curve_id: int
    segments: tuple[Segment, ...]
    primitive: PrimitiveShape | None


@dataclass(frozen=True)
class VectorDoc:
    width: int
    height: int
    paths: tuple[Path, ...]
    primitives: tuple[Primitive, ...]
    layer_mode: Literal["stacked", "cutout"]
    precision: int = 2  # decimal places used when serialising numbers


@dataclass(frozen=True)
class StageReport:
    stage: str
    duration_s: float
    params: Mapping[str, object]
    debug_images: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class VectorizeResult:
    doc: VectorDoc
    reports: tuple[StageReport, ...]
