"""Export stage: assemble a VectorDoc from fitted curves and serialise it to SVG."""
from __future__ import annotations

import math
import zlib

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
    Fill,
    GradientStop,
    LinearGradient,
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


def _scale_fill(fill: Fill, k: float) -> Fill:
    if isinstance(fill, LinearGradient):
        return replace(
            fill,
            x1=fill.x1 * k,
            y1=fill.y1 * k,
            x2=fill.x2 * k,
            y2=fill.y2 * k,
        )
    return fill


def _to_original_units(
    paths: list[Path], primitives: list[Primitive], scale: float
) -> tuple[tuple[Path, ...], tuple[Primitive, ...]]:
    """The single place where processing-pixel geometry is divided by `scale`."""
    if scale == 1.0:
        return tuple(paths), tuple(primitives)
    k = 1.0 / scale
    return (
        tuple(
            replace(
                p,
                contours=tuple(tuple(_scale_segment(s, k) for s in c) for c in p.contours),
                fill=_scale_fill(p.fill, k),
            )
            for p in paths
        ),
        tuple(
            replace(p, shape=_scale_shape(p.shape, k), fill=_scale_fill(p.fill, k))
            for p in primitives
        ),
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
        if plan.region in getattr(regions, "gradients", {}):
            fill = regions.gradients[plan.region]
        else:
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


def _hex_rgb(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hex(fill: SolidFill) -> str:
    return _hex_rgb(fill.rgb)


def _fill_attr(fill: Fill) -> str:
    if isinstance(fill, LinearGradient):
        return f'fill="url(#{fill.id})"'
    return f'fill="{_hex(fill)}"'


def _path_element(path: Path, doc: VectorDoc) -> str:
    d = "".join(_contour_d(c, doc.precision) for c in path.contours)
    holed = doc.layer_mode == "cutout" or len(path.contours) > 1
    rule = ' fill-rule="evenodd"' if holed else ""
    return f'<path d={quoteattr(d)} {_fill_attr(path.fill)}{rule}/>'


def _rotation(angle_deg: float, x: float, y: float, n: int) -> str:
    if not angle_deg:
        return ""
    return f' transform="rotate({_num(angle_deg, n)} {_num(x, n)} {_num(y, n)})"'


def _primitive_element(prim: Primitive, n: int) -> str:
    shape, fill = prim.shape, _fill_attr(prim.fill)
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


def _gradient_def(grad: LinearGradient, n: int) -> str:
    stops = "".join(
        f'<stop offset="{_num(s.offset * 100, n)}%" stop-color="{_hex_rgb(s.rgb)}"'
        + (f' stop-opacity="{_num(s.opacity, n)}"' if s.opacity < 1.0 else "")
        + "/>"
        for s in grad.stops
    )
    return (
        f'<linearGradient id="{grad.id}" '
        f'x1="{_num(grad.x1, n)}" y1="{_num(grad.y1, n)}" '
        f'x2="{_num(grad.x2, n)}" y2="{_num(grad.y2, n)}" '
        f'gradientUnits="userSpaceOnUse">{stops}</linearGradient>'
    )


def to_svg(doc: VectorDoc) -> str:
    seen_grads: dict[str, LinearGradient] = {}
    for p in doc.paths:
        if isinstance(p.fill, LinearGradient):
            seen_grads[p.fill.id] = p.fill
    for prim in doc.primitives:
        if isinstance(prim.fill, LinearGradient):
            seen_grads[prim.fill.id] = prim.fill

    defs = ""
    if seen_grads:
        defs = f"<defs>{''.join(_gradient_def(g, doc.precision) for g in seen_grads.values())}</defs>"

    items = [
        (p.z_order, _path_element(p, doc))
        for p in doc.paths
        if any(p.contours) and all(c for c in p.contours)
    ]
    items += [(p.z_order, _primitive_element(p, doc.precision)) for p in doc.primitives]
    body = "".join(el for _, el in sorted(items, key=lambda item: item[0]))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{doc.width}" height="{doc.height}" '
        f'viewBox="0 0 {doc.width} {doc.height}">{defs}{body}</svg>'
    )


def _arc_to_beziers(arc: Arc) -> list[CubicBezier]:
    cx, cy = arc.center
    r = arc.radius
    if r <= 0:
        return [CubicBezier(arc.p0, arc.p0, arc.p1, arc.p1)]
    
    th0 = math.atan2(arc.p0[1] - cy, arc.p0[0] - cx)
    th1 = math.atan2(arc.p1[1] - cy, arc.p1[0] - cx)
    dth = th1 - th0
    if arc.sweep and dth < 0:
        dth += 2 * math.pi
    elif not arc.sweep and dth > 0:
        dth -= 2 * math.pi
        
    num_segs = max(1, math.ceil(abs(dth) / (math.pi / 2)))
    step = dth / num_segs
    beziers = []
    
    cur_th = th0
    for i in range(num_segs):
        next_th = cur_th + step
        k = (4.0 / 3.0) * math.tan(step / 4.0)
        p0 = arc.p0 if i == 0 else (cx + r * math.cos(cur_th), cy + r * math.sin(cur_th))
        p3 = arc.p1 if i == num_segs - 1 else (cx + r * math.cos(next_th), cy + r * math.sin(next_th))
        c1 = (p0[0] - k * r * math.sin(cur_th), p0[1] + k * r * math.cos(cur_th))
        c2 = (p3[0] + k * r * math.sin(next_th), p3[1] - k * r * math.cos(next_th))
        beziers.append(CubicBezier(p0, c1, c2, p3))
        cur_th = next_th
        
    return beziers


def to_pdf(doc: VectorDoc) -> bytes:
    """Export a VectorDoc directly to a valid, standalone, compressed vector PDF 1.4."""
    w, h = doc.width, doc.height
    prec = max(2, doc.precision)

    def py(y: float) -> float:
        return h - y

    def fmt(v: float) -> str:
        s = f"{v:.{prec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s

    stream_ops: list[str] = []
    shadings: list[dict] = []
    shading_map: dict[int, str] = {}

    # Sort elements by z_order
    elements = [(p.z_order, p) for p in doc.paths] + [(p.z_order, p) for p in doc.primitives]
    elements.sort(key=lambda t: t[0])

    for _, elem in elements:
        fill = elem.fill
        is_gradient = isinstance(fill, LinearGradient)
        sh_name = None

        if is_gradient:
            grad_id = id(fill)
            if grad_id not in shading_map:
                sh_name = f"Sh{len(shadings) + 1}"
                shading_map[grad_id] = sh_name
                # LinearGradient coords in PDF space:
                # PDF y is inverted
                c0 = [round(c / 255.0, 4) for c in fill.stops[0].rgb]
                c1 = [round(c / 255.0, 4) for c in fill.stops[-1].rgb]
                shadings.append({
                    "name": sh_name,
                    "x1": fill.x1,
                    "y1": py(fill.y1),
                    "x2": fill.x2,
                    "y2": py(fill.y2),
                    "c0": c0,
                    "c1": c1,
                })
            else:
                sh_name = shading_map[grad_id]

        if is_gradient:
            stream_ops.append("q")  # push graphics state for clip
        else:
            r, g, b = [c / 255.0 for c in fill.rgb]
            stream_ops.append(f"{fmt(r)} {fmt(g)} {fmt(b)} rg")

        if isinstance(elem, Path):
            for contour in elem.contours:
                if not contour:
                    continue
                p0 = contour[0].p0
                stream_ops.append(f"{fmt(p0[0])} {fmt(py(p0[1]))} m")
                for seg in contour:
                    if isinstance(seg, Line):
                        stream_ops.append(f"{fmt(seg.p1[0])} {fmt(py(seg.p1[1]))} l")
                    elif isinstance(seg, CubicBezier):
                        c1 = seg.c1
                        c2 = seg.c2
                        p1 = seg.p1
                        stream_ops.append(
                            f"{fmt(c1[0])} {fmt(py(c1[1]))} {fmt(c2[0])} {fmt(py(c2[1]))} {fmt(p1[0])} {fmt(py(p1[1]))} c"
                        )
                    elif isinstance(seg, Arc):
                        for bez in _arc_to_beziers(seg):
                            c1 = bez.c1
                            c2 = bez.c2
                            p1 = bez.p1
                            stream_ops.append(
                                f"{fmt(c1[0])} {fmt(py(c1[1]))} {fmt(c2[0])} {fmt(py(c2[1]))} {fmt(p1[0])} {fmt(py(p1[1]))} c"
                            )
                stream_ops.append("h")
        elif isinstance(elem, Primitive):
            shape = elem.shape
            if isinstance(shape, Rect):
                if not shape.angle_deg:
                    stream_ops.append(f"{fmt(shape.x)} {fmt(py(shape.y + shape.h))} {fmt(shape.w)} {fmt(shape.h)} re")
                else:
                    rad = math.radians(shape.angle_deg)
                    cos_a, sin_a = math.cos(rad), math.sin(rad)
                    p0 = (shape.x, shape.y)
                    p1 = (shape.x + shape.w * cos_a, shape.y + shape.w * sin_a)
                    p2 = (shape.x + shape.w * cos_a - shape.h * sin_a, shape.y + shape.w * sin_a + shape.h * cos_a)
                    p3 = (shape.x - shape.h * sin_a, shape.y + shape.h * cos_a)
                    stream_ops.append(f"{fmt(p0[0])} {fmt(py(p0[1]))} m {fmt(p1[0])} {fmt(py(p1[1]))} l {fmt(p2[0])} {fmt(py(p2[1]))} l {fmt(p3[0])} {fmt(py(p3[1]))} l h")
            elif isinstance(shape, (Circle, Ellipse)):
                cx, cy = shape.cx, shape.cy
                rx = shape.r if isinstance(shape, Circle) else shape.rx
                ry = shape.r if isinstance(shape, Circle) else shape.ry
                rot = getattr(shape, "angle_deg", 0.0)
                kappa = 0.5522847498307936
                pts = [
                    ((cx + rx, cy), (cx + rx, cy + kappa * ry), (cx + kappa * rx, cy + ry), (cx, cy + ry)),
                    ((cx, cy + ry), (cx - kappa * rx, cy + ry), (cx - rx, cy + kappa * ry), (cx - rx, cy)),
                    ((cx - rx, cy), (cx - rx, cy - kappa * ry), (cx - kappa * rx, cy - ry), (cx, cy - ry)),
                    ((cx, cy - ry), (cx + kappa * rx, cy - ry), (cx + rx, cy - kappa * ry), (cx + rx, cy)),
                ]
                rad = math.radians(rot)
                cos_r, sin_r = math.cos(rad), math.sin(rad)
                def rot_pt(pt: Point) -> Point:
                    dx, dy = pt[0] - cx, pt[1] - cy
                    return (cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r)

                first = rot_pt(pts[0][0])
                stream_ops.append(f"{fmt(first[0])} {fmt(py(first[1]))} m")
                for p0, c1, c2, p1 in pts:
                    rc1 = rot_pt(c1)
                    rc2 = rot_pt(c2)
                    rp1 = rot_pt(p1)
                    stream_ops.append(f"{fmt(rc1[0])} {fmt(py(rc1[1]))} {fmt(rc2[0])} {fmt(py(rc2[1]))} {fmt(rp1[0])} {fmt(py(rp1[1]))} c")
                stream_ops.append("h")
            elif isinstance(shape, Polygon):
                if shape.points:
                    p0 = shape.points[0]
                    stream_ops.append(f"{fmt(p0[0])} {fmt(py(p0[1]))} m")
                    for pt in shape.points[1:]:
                        stream_ops.append(f"{fmt(pt[0])} {fmt(py(pt[1]))} l")
                    stream_ops.append("h")

        if is_gradient:
            # Clip to path (non-zero winding) and paint shading
            stream_ops.append("W n")
            stream_ops.append(f"/{sh_name} sh")
            stream_ops.append("Q")  # pop graphics state (restores clip)
        else:
            # Fill path (even-odd winding rule matches SVG fill-rule=evenodd)
            stream_ops.append("f*")

    content_data = "\n".join(stream_ops).encode("latin-1")
    compressed = zlib.compress(content_data)

    # Build PDF object hierarchy
    objects: list[bytes] = []

    # Obj 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # Obj 2: Pages
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")

    # Shading objects if any
    shading_entries = []
    shading_obj_nums = {}
    next_obj = 5
    for sh in shadings:
        sh_num = next_obj
        shading_obj_nums[sh["name"]] = sh_num
        shading_entries.append(f"/{sh['name']} {sh_num} 0 R")
        next_obj += 1

    shading_res_str = f"/Shading << {' '.join(shading_entries)} >>" if shading_entries else ""

    # Obj 3: Page
    page_dict = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Contents 4 0 R /Resources << {shading_res_str} >> >>".encode("latin-1")
    objects.append(page_dict)

    # Obj 4: Content stream
    stream_dict = f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("latin-1") + compressed + b"\nendstream"
    objects.append(stream_dict)

    # Add each Shading object
    for sh in shadings:
        c0_s = " ".join(str(c) for c in sh["c0"])
        c1_s = " ".join(str(c) for c in sh["c1"])
        sh_dict = (
            f"<< /ShadingType 2 /ColorSpace /DeviceRGB "
            f"/Coords [{fmt(sh['x1'])} {fmt(sh['y1'])} {fmt(sh['x2'])} {fmt(sh['y2'])}] "
            f"/Function << /FunctionType 2 /Domain [0 1] /C0 [{c0_s}] /C1 [{c1_s}] /N 1 >> "
            f"/Extend [true true] >>"
        ).encode("latin-1")
        objects.append(sh_dict)

    # Assemble complete PDF with cross-reference table
    out = bytearray()
    out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("latin-1"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

    out.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    return bytes(out)
