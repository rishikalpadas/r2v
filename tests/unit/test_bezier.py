import math

import numpy as np

from engine.geometry.bezier import fit_cubic


def _s_curve(n=40):
    t = np.linspace(0, 1, n)
    return [(float(40 * x), float(10 * math.sin(2 * math.pi * x))) for x in t]


def _eval(seg, t):
    s = 1 - t
    return tuple(
        s**3 * seg.p0[k] + 3 * s * s * t * seg.c1[k] + 3 * s * t * t * seg.c2[k] + t**3 * seg.p1[k]
        for k in (0, 1)
    )


def _dist_to_chain(chain, p):
    ts = np.linspace(0, 1, 400)
    return min(math.dist(p, _eval(seg, t)) for seg in chain for t in ts)


def test_fit_cubic_within_tolerance():
    pts = _s_curve()
    chain = fit_cubic(pts, 0.75)
    assert 1 <= len(chain) <= 2
    assert all(_dist_to_chain(chain, p) <= 0.75 for p in pts)


def test_fit_cubic_endpoints_and_connectivity_exact():
    pts = _s_curve()
    chain = fit_cubic(pts, 0.1)
    assert chain[0].p0 == pts[0] and chain[-1].p1 == pts[-1]
    for a, b in zip(chain, chain[1:]):
        assert a.p1 == b.p0


def test_fit_cubic_g1_joins():
    t = np.linspace(0, 1, 120)
    pts = [(float(100 * x), float(8 * math.sin(6 * math.pi * x))) for x in t]
    chain = fit_cubic(pts, 0.2)
    assert len(chain) >= 3
    for a, b in zip(chain, chain[1:]):
        v1 = (a.p1[0] - a.c2[0], a.p1[1] - a.c2[1])
        v2 = (b.c1[0] - b.p0[0], b.c1[1] - b.p0[1])
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
        assert math.degrees(math.acos(min(1.0, cos))) < 1.0


def test_fit_cubic_degenerate_inputs():
    assert fit_cubic([(1.0, 1.0)], 0.5) == ()
    two = fit_cubic([(0.0, 0.0), (6.0, 0.0)], 0.5)
    assert len(two) == 1 and two[0].p0 == (0.0, 0.0) and two[0].p1 == (6.0, 0.0)
    same = fit_cubic([(2.0, 2.0)] * 4, 0.5)
    assert len(same) == 1 and same[0].p0 == same[0].p1 == (2.0, 2.0)
