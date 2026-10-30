"""Export stage: assemble a VectorDoc from fitted curves and serialise it to SVG."""
from __future__ import annotations

from dataclasses import replace
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


def build_doc(
    regions: RegionMap,
    graph: BoundaryGraph,
    fitted: tuple[FittedCurve, ...],
    params: ExportParams,
) -> VectorDoc:
    h, w = regions.labels.shape
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
    paths: list[Path] = []
    primitives: list[Primitive] = []
    for z, region in enumerate(ordered):
        loops = _emitted_loops(region, graph, stacked, sides)
        fill = SolidFill(tuple(int(c) for c in regions.palette_rgb[region]))
        if stacked and z == 0 and _needs_canvas_underlay(loops[0], graph):
            paths.append(Path((_canvas_frame(w, h),), fill, z))
            continue
        shape = (
            _single_primitive(region, loops[0], by_id, stacked, curve_sides)
            if len(loops) == 1
            else None
        )
        if shape is not None:
            primitives.append(Primitive(shape, fill, z))
            continue
        contours = tuple(_loop_segments(loop, by_id) for loop in loops)
        paths.append(Path(contours, fill, z))
    return VectorDoc(
        width=w,
        height=h,
        paths=tuple(paths),
        primitives=tuple(primitives),
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
