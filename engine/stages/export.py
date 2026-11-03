"""Export stage: assemble a VectorDoc from fitted curves and serialise it to SVG."""
from __future__ import annotations

from dataclasses import dataclass, replace
from xml.sax.saxutils import quoteattr

from engine.presets import ExportParams
from engine.types import (
    OUTSIDE,
    TRANSPARENT,
    Arc,
    BoundaryGraph,
    Circle,
    CubicBezier,
    Ellipse,
    FittedCurve,
    Line,
    LoopRef,
    Path,
    Point,
    Polygon,
    Primitive,
    PrimitiveShape,
    Rect,
    RegionMap,
    Segment,
    SolidFill,
    VectorDoc,
)

# ---------------------------------------------------------------- document assembly


def _reverse_segment(seg: Segment) -> Segment:
    if isinstance(seg, Line):
        return Line(seg.p1, seg.p0)
    if isinstance(seg, Arc):
        return replace(seg, p0=seg.p1, p1=seg.p0, sweep=not seg.sweep)
    return CubicBezier(seg.p1, seg.c2, seg.c1, seg.p0)


def _loop_segments(loop: LoopRef, fitted: dict[int, FittedCurve]) -> tuple[Segment, ...]:
    out: list[Segment] = []
    for curve_id, is_reversed in loop:
        segs = fitted[curve_id].segments
        if is_reversed:
            out.extend(_reverse_segment(s) for s in reversed(segs))
        else:
            out.extend(segs)
    return tuple(out)


def _loop_area(loop: LoopRef, curves: dict[int, tuple[Point, ...]]) -> float:
    """Enclosed area of a loop: signed shoelace walked in loop order (reversed curves flipped)."""
    total = 0.0
    for curve_id, is_reversed in loop:
        pts = curves[curve_id][::-1] if is_reversed else curves[curve_id]
        total += sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    return abs(total) / 2.0


def _single_primitive(
    region: int,
    loop: LoopRef,
    fitted: dict[int, FittedCurve],
    stacked: bool,
    curves: dict[int, tuple[int, int]],
) -> PrimitiveShape | None:
    """The primitive replacing a one-curve loop, if allowed.

    In cutout mode a neighbouring region's hole is drawn from the curve's fitted segments, so a
    primitive (a different shape) would leave slivers; only allow it against transparency or the
    canvas edge, where nothing else shares the edge.
    """
    if len(loop) != 1:
        return None
    curve_id = loop[0][0]
    if not stacked:
        left, right = curves[curve_id]
        other = right if left == region else left
        if other not in (TRANSPARENT, OUTSIDE):
            return None
    return fitted[curve_id].primitive


def _borders_transparent(loop: LoopRef, sides: dict[int, int]) -> bool:
    return any(sides[curve_id] == TRANSPARENT for curve_id, _ in loop)


def _emitted_loops(
    region: int, graph: BoundaryGraph, stacked: bool, sides: dict[int, int]
) -> tuple[LoopRef, ...]:
    """Loops to draw. Stacked keeps the outer loop plus holes open to transparency; holes that
    border other regions are omitted because those regions paint over them."""
    loops = graph.region_loops[region]
    if not stacked:
        return loops
    return (loops[0], *(h for h in loops[1:] if _borders_transparent(h, sides)))


def _canvas_frame(w: int, h: int) -> tuple[Segment, ...]:
    return (Line((0, 0), (w, 0)), Line((w, 0), (w, h)), Line((w, h), (0, h)), Line((0, h), (0, 0)))


def _needs_canvas_underlay(loop: LoopRef, graph: BoundaryGraph) -> bool:
    """True when the bottom layer must be widened to the canvas: its outer loop is not the
    canvas frame and nothing is transparent.

    A shape touching the canvas edge makes the bottom region's outer loop walk around it, so no
    layer lies under that shape. Fitted (non pixel-aligned) shared edges would then be
    anti-aliased from both sides and leave partly transparent seams. Every other region is
    painted on top, so covering the whole canvas is exact.
    """
    if any(TRANSPARENT in (c.left, c.right) for c in graph.curves):
        return False
    sides = {c.id: (c.left, c.right) for c in graph.curves}
    return any(OUTSIDE not in sides[curve_id] for curve_id, _ in loop)


def _scale_point(p: Point, k: float) -> Point:
    return (p[0] * k, p[1] * k)


def _scale_segment(seg: Segment, k: float) -> Segment:
    if isinstance(seg, Line):
        return Line(_scale_point(seg.p0, k), _scale_point(seg.p1, k))
    if isinstance(seg, Arc):
        return replace(
            seg,
            p0=_scale_point(seg.p0, k),
            p1=_scale_point(seg.p1, k),
            center=_scale_point(seg.center, k),
            radius=seg.radius * k,
        )
    return CubicBezier(*(_scale_point(p, k) for p in (seg.p0, seg.c1, seg.c2, seg.p1)))


def _scale_shape(shape: PrimitiveShape, k: float) -> PrimitiveShape:
    if isinstance(shape, Circle):
        return Circle(shape.cx * k, shape.cy * k, shape.r * k)
    if isinstance(shape, Ellipse):
        return replace(shape, cx=shape.cx * k, cy=shape.cy * k, rx=shape.rx * k, ry=shape.ry * k)
    if isinstance(shape, Rect):
        return replace(shape, x=shape.x * k, y=shape.y * k, w=shape.w * k, h=shape.h * k)
    return Polygon(tuple(_scale_point(p, k) for p in shape.points))


def _to_original_units(
    paths: list[Path], primitives: list[Primitive], scale: float
) -> tuple[tuple[Path, ...], tuple[Primitive, ...]]:
    """The single place where processing-pixel geometry is divided by `scale`."""
    if scale == 1.0:
        return tuple(paths), tuple(primitives)
    k = 1.0 / scale
    return (
        tuple(
            replace(p, contours=tuple(tuple(_scale_segment(s, k) for s in c) for c in p.contours))
            for p in paths
        ),
        tuple(replace(p, shape=_scale_shape(p.shape, k)) for p in primitives),
    )


@dataclass(frozen=True)
class LayerPlan:
    """One painted layer: `region` at `z` drawn from `loops`, as the canvas underlay, as the
    primitive of curve `primitive_curve`, or (neither) as a path of the loops' fitted segments."""

    z: int
    region: int
    loops: tuple[LoopRef, ...]
    underlay: bool
    primitive_curve: int | None


def plan_layers(
    regions: RegionMap,
    graph: BoundaryGraph,
    fitted: tuple[FittedCurve, ...],
    params: ExportParams,
) -> tuple[LayerPlan, ...]:
    """How `build_doc` draws each region, in paint order. Analysis tools use it to know which
    curves end up as primitives and which as path segments."""
    by_id = {f.curve_id: f for f in fitted}
    points = {c.id: c.points for c in graph.curves}
    sides = {c.id: c.right for c in graph.curves}
    curve_sides = {c.id: (c.left, c.right) for c in graph.curves}
    stacked = params.layer_mode == "stacked"

    # Paint larger enclosed areas first so a nested region always lands on top of its container
    # (pixel counts would mis-order a thin ring around a large disc).
    ordered = sorted(
        (r for r, loops in graph.region_loops.items() if loops and 0 <= r < len(regions.palette_rgb)),
        key=lambda r: (-_loop_area(graph.region_loops[r][0], points), r),
    )
    plans: list[LayerPlan] = []
    for z, region in enumerate(ordered):
        loops = _emitted_loops(region, graph, stacked, sides)
        if stacked and z == 0 and _needs_canvas_underlay(loops[0], graph):
            plans.append(LayerPlan(z, region, loops, True, None))
            continue
        shape = (
            _single_primitive(region, loops[0], by_id, stacked, curve_sides)
            if len(loops) == 1
            else None
        )
        curve = loops[0][0][0] if shape is not None else None
        plans.append(LayerPlan(z, region, loops, False, curve))
    return tuple(plans)


def build_doc(
    regions: RegionMap,
    graph: BoundaryGraph,
    fitted: tuple[FittedCurve, ...],
    params: ExportParams,
    scale: float = 1.0,
    original_size: tuple[int, int] | None = None,
) -> VectorDoc:
    """Assemble the document. Inputs are in processing pixels (original pixels x `scale`); the
    document is in original pixels with size `original_size` (w, h), which defaults to the label
    grid's size."""
    w, h = original_size or (regions.labels.shape[1], regions.labels.shape[0])
    by_id = {f.curve_id: f for f in fitted}
    paths: list[Path] = []
    primitives: list[Primitive] = []
    for plan in plan_layers(regions, graph, fitted, params):
        fill = SolidFill(tuple(int(c) for c in regions.palette_rgb[plan.region]))
        if plan.underlay:
            paths.append(Path((_canvas_frame(w * scale, h * scale),), fill, plan.z))
        elif plan.primitive_curve is not None:
            primitives.append(Primitive(by_id[plan.primitive_curve].primitive, fill, plan.z))
        else:
            contours = tuple(_loop_segments(loop, by_id) for loop in plan.loops)
            paths.append(Path(contours, fill, plan.z))
    out_paths, out_primitives = _to_original_units(paths, primitives, scale)
    return VectorDoc(
        width=w,
        height=h,
        paths=out_paths,
        primitives=out_primitives,
        layer_mode=params.layer_mode,
        precision=params.precision,
    )


# ---------------------------------------------------------------- SVG serialisation


def _num(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _pt(p: Point, precision: int) -> str:
    return f"{_num(p[0], precision)} {_num(p[1], precision)}"


def _segment_cmd(seg: Segment, n: int) -> str:
    if isinstance(seg, Line):
        return f"L{_pt(seg.p1, n)}"
    if isinstance(seg, Arc):
        r = _num(seg.radius, n)
        return f"A{r} {r} 0 {int(seg.large_arc)} {int(seg.sweep)} {_pt(seg.p1, n)}"
    return f"C{_pt(seg.c1, n)} {_pt(seg.c2, n)} {_pt(seg.p1, n)}"


def _contour_d(contour: tuple[Segment, ...], n: int) -> str:
    if not contour:
        return ""
    cmds = [f"M{_pt(contour[0].p0, n)}"]
    last = contour[-1]
    for seg in contour:
        if seg is last and isinstance(seg, Line) and seg.p1 == contour[0].p0:
            continue  # Z draws the closing line
        cmds.append(_segment_cmd(seg, n))
    cmds.append("Z")
    return "".join(cmds)


def _hex(fill: SolidFill) -> str:
    return "#{:02x}{:02x}{:02x}".format(*fill.rgb)


def _path_element(path: Path, doc: VectorDoc) -> str:
    d = "".join(_contour_d(c, doc.precision) for c in path.contours)
    holed = doc.layer_mode == "cutout" or len(path.contours) > 1
    rule = ' fill-rule="evenodd"' if holed else ""
    return f'<path d={quoteattr(d)} fill="{_hex(path.fill)}"{rule}/>'


def _rotation(angle_deg: float, x: float, y: float, n: int) -> str:
    if not angle_deg:
        return ""
    return f' transform="rotate({_num(angle_deg, n)} {_num(x, n)} {_num(y, n)})"'


def _primitive_element(prim: Primitive, n: int) -> str:
    shape, fill = prim.shape, f'fill="{_hex(prim.fill)}"'
    if isinstance(shape, Circle):
        return (f'<circle cx="{_num(shape.cx, n)}" cy="{_num(shape.cy, n)}" '
                f'r="{_num(shape.r, n)}" {fill}/>')
    if isinstance(shape, Ellipse):
        rot = _rotation(shape.angle_deg, shape.cx, shape.cy, n)
        return (f'<ellipse cx="{_num(shape.cx, n)}" cy="{_num(shape.cy, n)}" '
                f'rx="{_num(shape.rx, n)}" ry="{_num(shape.ry, n)}" {fill}{rot}/>')
    if isinstance(shape, Rect):
        rot = _rotation(shape.angle_deg, shape.x, shape.y, n)
        return (f'<rect x="{_num(shape.x, n)}" y="{_num(shape.y, n)}" '
                f'width="{_num(shape.w, n)}" height="{_num(shape.h, n)}" {fill}{rot}/>')
    pts = " ".join(_pt(p, n) for p in shape.points)
    return f'<polygon points="{pts}" {fill}/>'


def to_svg(doc: VectorDoc) -> str:
    items = [
        (p.z_order, _path_element(p, doc))
        for p in doc.paths
        if any(p.contours) and all(c for c in p.contours)
    ]
    items += [(p.z_order, _primitive_element(p, doc.precision)) for p in doc.primitives]
    body = "".join(el for _, el in sorted(items, key=lambda item: item[0]))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{doc.width}" height="{doc.height}" '
        f'viewBox="0 0 {doc.width} {doc.height}">{body}</svg>'
    )
