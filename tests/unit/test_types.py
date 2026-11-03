import dataclasses

import pytest

from engine.errors import StageError
from engine.types import OUTSIDE, TRANSPARENT, Line


def test_dataclasses_are_frozen():
    line = Line((0, 0), (1, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        line.p0 = (2, 2)


def test_stage_error_message():
    err = StageError("segment", "boom")
    assert str(err) == "segment: boom"
    assert err.stage == "segment"


def test_label_constants():
    assert TRANSPARENT == -1 and OUTSIDE == -2


def test_stage_error_round_trips_pickle():
    import pickle

    err = pickle.loads(pickle.dumps(StageError("fit", "boom")))
    assert isinstance(err, StageError)
    assert (err.stage, err.reason, str(err)) == ("fit", "boom", "fit: boom")


def test_array_holding_types_compare_by_identity():
    import numpy as np

    from engine.types import RasterImage, RegionMap, StageReport

    def make():
        return (
            RasterImage(np.zeros((2, 2, 4), np.uint8), (2, 2)),
            RegionMap(np.zeros((2, 2), np.int32), np.zeros((1, 3), np.uint8), np.zeros((1, 3)), frozenset()),
            StageReport("segment", 0.0, {}, {"x": np.zeros((2, 2))}),
        )

    for a, b in zip(make(), make()):
        assert a == a
        assert not (a == b)  # no element-wise ndarray comparison
        hash(a)


@pytest.mark.parametrize("scale", [1, 1.0, 2.5, 4, 0.5])
def test_raster_image_accepts_positive_finite_scale(scale):
    import numpy as np

    from engine.types import RasterImage

    assert RasterImage(np.zeros((2, 2, 4), np.uint8), (2, 2), scale=scale).scale == scale


def test_raster_image_accepts_numpy_scale():
    import numpy as np

    from engine.types import RasterImage

    assert RasterImage(np.zeros((2, 2, 4), np.uint8), (2, 2), scale=np.float64(2.0)).scale == 2.0


@pytest.mark.parametrize(
    "scale", [0, 0.0, -1.0, float("nan"), float("inf"), -float("inf"), True, False, "4", None, 1j]
)
def test_raster_image_rejects_bad_scale(scale):
    import numpy as np

    from engine.types import RasterImage

    with pytest.raises(ValueError, match="scale"):
        RasterImage(np.zeros((2, 2, 4), np.uint8), (2, 2), scale=scale)
