"""Pipeline: run the preset's stages in order and produce a VectorizeResult."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace

from engine.errors import InputError, StageError
from engine.loader import load_image
from engine.presets import Preset, apply_overrides, resolve_preset
from engine.stages.export import build_doc
from engine.stages.fit_geometry import fit_geometry
from engine.stages.segment import segment
from engine.stages.trace import trace
from engine.types import (
    BoundaryGraph,
    FittedCurve,
    RasterImage,
    RegionMap,
    StageReport,
    VectorDoc,
    VectorizeResult,
)


@dataclass(frozen=True)
class PipelineState:
    image: RasterImage
    preset: Preset
    regions: RegionMap | None = None
    graph: BoundaryGraph | None = None
    fitted: tuple[FittedCurve, ...] | None = None
    doc: VectorDoc | None = None


def _need(value, what: str, stage: str):
    if value is None:
        raise StageError(stage, f"missing {what}; stage order is invalid")
    return value


def _run_segment(state: PipelineState) -> PipelineState:
    return replace(state, regions=segment(state.image, state.preset.segment))


def _run_trace(state: PipelineState) -> PipelineState:
    return replace(state, graph=trace(_need(state.regions, "regions", "trace")))


def _run_fit(state: PipelineState) -> PipelineState:
    graph = _need(state.graph, "boundary graph", "fit")
    return replace(state, fitted=fit_geometry(graph, state.preset.fit))


def _run_export(state: PipelineState) -> PipelineState:
    doc = build_doc(
        _need(state.regions, "regions", "export"),
        _need(state.graph, "boundary graph", "export"),
        _need(state.fitted, "fitted curves", "export"),
        state.preset.export,
    )
    return replace(state, doc=doc)


# Stage name -> runner. Later tasks swap entries (e.g. "fit").
STAGES: Mapping[str, Callable[[PipelineState], PipelineState]] = {
    "segment": _run_segment,
    "trace": _run_trace,
    "fit": _run_fit,
    "export": _run_export,
}


def _stage_params(stage: str, preset: Preset) -> dict[str, object]:
    group = getattr(preset, stage, None)
    return asdict(group) if is_dataclass(group) else {}


def _run_stage(name: str, state: PipelineState) -> PipelineState:
    func = STAGES.get(name)
    if func is None:
        raise StageError(name, "unknown stage")
    try:
        return func(state)
    except (InputError, StageError):
        raise
    except Exception as exc:  # noqa: BLE001 - any other failure is reported as a StageError
        raise StageError(name, f"internal error ({type(exc).__name__})") from exc


def vectorize(
    image_bytes: bytes,
    preset: str = "auto",
    overrides: Mapping[str, object] | None = None,
) -> VectorizeResult:
    resolved = resolve_preset(preset)
    if overrides:
        resolved = apply_overrides(resolved, overrides)
    state = PipelineState(image=load_image(image_bytes), preset=resolved)
    reports: list[StageReport] = []
    for name in resolved.stages:
        start = time.perf_counter()
        state = _run_stage(name, state)
        reports.append(
            StageReport(
                stage=name,
                duration_s=time.perf_counter() - start,
                params=_stage_params(name, resolved),
                debug_images={},
            )
        )
    if state.doc is None:
        raise StageError("export", "pipeline produced no document")
    return VectorizeResult(doc=state.doc, reports=tuple(reports))
