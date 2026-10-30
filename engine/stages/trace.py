"""Trace stage: build the shared-boundary graph of a RegionMap."""
from __future__ import annotations

from engine.errors import StageError
from engine.geometry.stitching import build_boundary_graph
from engine.types import BoundaryGraph, RegionMap


def trace(regions: RegionMap) -> BoundaryGraph:
    try:
        return build_boundary_graph(regions)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure here is a trace-stage failure
        raise StageError("trace", f"internal error ({type(exc).__name__})") from exc
