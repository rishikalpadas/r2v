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
