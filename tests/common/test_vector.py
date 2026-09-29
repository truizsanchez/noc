import math
import random

import pytest

from noc.common.vector import Vector


def test_arithmetic_returns_new_vectors() -> None:
    v, u = Vector(0, 0), Vector(4, 5)
    w = v + u
    assert w == Vector(4, 5)
    assert v == Vector(0, 0)  # unchanged: no static/nonstatic distinction needed
    assert u - Vector(1, 1) == Vector(3, 4)
    assert u * 2 == 2 * u == Vector(8, 10)
    assert u / 2 == Vector(2, 2.5)
    assert -u == Vector(-4, -5)


def test_exercise_1_7() -> None:
    v = Vector(1, 5)
    u = v * 2
    w = (v - u) / 3
    assert w.xy == pytest.approx((-1 / 3, -5 / 3))


def test_unpacks_like_a_point() -> None:
    x, y = Vector(3, 4)
    assert (x, y) == (3, 4)
    assert not Vector()
    assert Vector(0, 1)


def test_magnitude_and_normalize() -> None:
    v = Vector(3, 4)
    assert v.mag() == 5
    assert v.mag_sq() == 25
    assert v.normalize().xy == pytest.approx((0.6, 0.8))
    assert Vector().normalize() == Vector()


def test_limit_and_set_mag() -> None:
    v = Vector(30, 40)
    assert v.limit(10).xy == pytest.approx((6, 8))
    assert Vector(3, 4).limit(10) == Vector(3, 4)
    assert v.set_mag(1).mag() == pytest.approx(1)
    assert v.limit(math.inf) == v


def test_heading_rotation_and_angles() -> None:
    assert Vector(0, 1).heading() == pytest.approx(math.pi / 2)
    assert Vector.from_angle(math.pi / 2, 2).xy == pytest.approx((0, 2))
    assert Vector(1, 0).rotate(math.pi / 2).xy == pytest.approx((0, 1))
    assert Vector(1, 0).angle_between(Vector(-1, 0)) == pytest.approx(math.pi)
    assert Vector(1, 0).angle_between(Vector()) == 0


def test_products_and_distances() -> None:
    assert Vector(1, 2).dot(Vector(3, 4)) == 11
    assert Vector(1, 1).dist(Vector(4, 5)) == 5
    assert Vector(0, 0).lerp(Vector(10, 20), 0.25) == Vector(2.5, 5)


def test_random2d_is_a_unit_vector() -> None:
    rng = random.Random(1)
    vectors = [Vector.random2d(rng) for _ in range(100)]
    assert all(v.mag() == pytest.approx(1) for v in vectors)
    assert len({round(v.heading(), 3) for v in vectors}) > 90
