"""Pipeline: run the preset's stages in order and produce a VectorizeResult."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace

from engine.errors import InputError, StageError
from engine.geometry.subpixel import make_edge_context
from engine.loader import load_image
from engine.presets import Preset, apply_overrides, resolve_preset
from engine.stages.enhance import enhance
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
    report: Mapping[str, object] | None = None  # the last stage's extra report params


def _need(value, what: str, stage: str):
    if value is None:
        raise StageError(stage, f"missing {what}; stage order is invalid")
    return value


def _run_enhance(state: PipelineState) -> PipelineState:
    # segment's max_colors decides exact-colour input, which enhance passes through unchanged
    image, report = enhance(state.image, state.preset.enhance, state.preset.segment.max_colors)
    return replace(state, image=image, report=report)


def _run_segment(state: PipelineState) -> PipelineState:
    return replace(state, regions=segment(state.image, state.preset.segment))


def _run_trace(state: PipelineState) -> PipelineState:
    return replace(state, graph=trace(_need(state.regions, "regions", "trace")))


def _run_fit(state: PipelineState) -> PipelineState:
    graph = _need(state.graph, "boundary graph", "fit")
    params = state.preset.fit
    context = None
    if params.enabled and params.subpixel:
        context = make_edge_context(state.image, _need(state.regions, "regions", "fit"))
    return replace(state, fitted=fit_geometry(graph, params, state.image.scale, context))


def _run_export(state: PipelineState) -> PipelineState:
    doc = build_doc(
        _need(state.regions, "regions", "export"),
        _need(state.graph, "boundary graph", "export"),
        _need(state.fitted, "fitted curves", "export"),
        state.preset.export,
        scale=state.image.scale,
        original_size=state.image.original_size,
    )
    return replace(state, doc=doc)


# Stage name -> runner. Later tasks swap entries (e.g. "fit").
STAGES: Mapping[str, Callable[[PipelineState], PipelineState]] = {
    "enhance": _run_enhance,
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
    return run_pipeline(load_image(image_bytes), resolved)


def run_stages(
    image: RasterImage, preset: Preset
) -> tuple[PipelineState, tuple[StageReport, ...]]:
    """Run `preset`'s stages and return the final state (regions, graph, fitted curves, doc)
    with the per-stage reports. `run_pipeline` is this plus the document check; analysis tools
    (e.g. `eval.defects`) use it to see the intermediate geometry behind a document."""
    state = PipelineState(image=image, preset=preset)
    reports: list[StageReport] = []
    for name in preset.stages:
        start = time.perf_counter()
        state = _run_stage(name, state)
        duration = time.perf_counter() - start
        params = {**_stage_params(name, preset), **(state.report or {})}
        state = replace(state, report=None)
        reports.append(
            StageReport(stage=name, duration_s=duration, params=params, debug_images={})
        )
    return state, tuple(reports)


def run_pipeline(image: RasterImage, preset: Preset) -> VectorizeResult:
    """Run `preset`'s stages on an already loaded image."""
    state, reports = run_stages(image, preset)
    if state.doc is None:
        raise StageError("export", "pipeline produced no document")
    return VectorizeResult(doc=state.doc, reports=reports)
