"""Export stage: assemble a VectorDoc from fitted curves and serialise it to SVG."""
from __future__ import annotations

from dataclasses import replace
from xml.sax.saxutils import quoteattr

from engine.presets import ExportParams
from engine.types import (
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
    """Absolute shoelace area enclosed by a loop of boundary curves."""
    total = 0.0
    for curve_id, _ in loop:
        pts = curves[curve_id]
        total += sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    return abs(total) / 2.0


def _single_primitive(loop: LoopRef, fitted: dict[int, FittedCurve]) -> PrimitiveShape | None:
    if len(loop) != 1:
        return None
    return fitted[loop[0][0]].primitive


def build_doc(
    regions: RegionMap,
    graph: BoundaryGraph,
    fitted: tuple[FittedCurve, ...],
    params: ExportParams,
) -> VectorDoc:
    h, w = regions.labels.shape
    by_id = {f.curve_id: f for f in fitted}
    points = {c.id: c.points for c in graph.curves}
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
        loops = graph.region_loops[region][:1] if stacked else graph.region_loops[region]
        fill = SolidFill(tuple(int(c) for c in regions.palette_rgb[region]))
        shape = _single_primitive(loops[0], by_id) if len(loops) == 1 else None
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
    rule = ' fill-rule="evenodd"' if doc.layer_mode == "cutout" else ""
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
    items = [(p.z_order, _path_element(p, doc)) for p in doc.paths]
    items += [(p.z_order, _primitive_element(p, doc.precision)) for p in doc.primitives]
    body = "".join(el for _, el in sorted(items, key=lambda item: item[0]))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{doc.width}" height="{doc.height}" '
        f'viewBox="0 0 {doc.width} {doc.height}">{body}</svg>'
    )
