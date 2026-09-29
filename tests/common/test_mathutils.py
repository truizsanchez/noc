import pytest

from noc.common.mathutils import clamp, remap


def test_remap_like_p5_map() -> None:
    assert remap(0.5, 0, 1, 0, 640) == 320
    assert remap(25, 0, 100, 100, 0) == 75
    assert remap(2, 0, 1, 0, 10) == 20  # no clamping, as in p5


def test_clamp() -> None:
    assert clamp(-3, 0, 10) == 0
    assert clamp(3.5, 0, 10) == pytest.approx(3.5)
    assert clamp(30, 0, 10) == 10
