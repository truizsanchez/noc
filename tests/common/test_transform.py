import math

import pytest

from noc.common.transform import IDENTITY, Transform


def close(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a == pytest.approx(b, abs=1e-9)


def test_identity_leaves_points_alone() -> None:
    assert IDENTITY.apply(3, -4) == (3, -4)
    assert IDENTITY.is_identity


def test_translate_then_rotate_like_p5() -> None:
    # p5: translate(100, 50); rotate(HALF_PI); point(10, 0) -> lands 10 px below the origin.
    t = IDENTITY.translated(100, 50).rotated(math.pi / 2)
    assert close(t.apply(10, 0), (100, 60))


def test_rotation_is_applied_before_a_later_translation() -> None:
    # rotate(HALF_PI); translate(10, 0): the translation happens along the rotated x axis.
    t = IDENTITY.rotated(math.pi / 2).translated(10, 0)
    assert close(t.apply(0, 0), (0, 10))


def test_scale_affects_later_translations() -> None:
    t = IDENTITY.scaled(0.5).translated(100, 0)
    assert close(t.apply(10, 0), (55, 0))
    assert t.scale == 0.5


def test_nested_transforms_compose_like_a_recursive_tree() -> None:
    # Each branch: translate(0, -len); rotate(a); scale(0.5) — the chapter 8 pattern.
    t = Transform(320, 240)
    for _ in range(3):
        t = t.translated(0, -40).rotated(math.pi / 6).scaled(0.5)
    x, y = t.apply(0, 0)
    assert y < 240
    assert x > 320
