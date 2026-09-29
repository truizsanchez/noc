import itertools
import random

import pytest

from noc.ch01_vectors.motion import (
    Mover,
    NoiseAcceleration,
    Train,
    random_acceleration,
    toward,
    toward_by_distance,
)
from noc.common.noise import Noise
from noc.common.vector import Vector


def test_motion_101() -> None:
    mover = Mover(Vector(0, 0), Vector(1, 0), Vector(0, 0.5))
    mover.update()
    assert mover.velocity == Vector(1, 0.5)
    assert mover.position == Vector(1, 0.5)
    mover.update()
    assert mover.position == Vector(2, 1.5)


def test_top_speed_limits_the_velocity() -> None:
    mover = Mover(Vector(), acceleration=Vector(0, 1), top_speed=5)
    for _ in range(20):
        mover.update()
    assert mover.velocity.mag() == pytest.approx(5)


def test_example_1_8_reaches_top_speed_eventually() -> None:
    mover = Mover(Vector(320, 120), acceleration=Vector(-0.001, 0.01), top_speed=10)
    for _ in range(2000):
        mover.update()
        mover.wrap_edges(640, 240)
        assert 0 <= mover.position.x <= 640
        assert 0 <= mover.position.y <= 240
    assert mover.velocity.mag() == pytest.approx(10)


def test_wrap_edges() -> None:
    mover = Mover(Vector(641, -1))
    mover.wrap_edges(640, 240)
    assert mover.position == Vector(0, 240)


def test_bounce_edges_reverses_only_the_axis_that_left() -> None:
    mover = Mover(Vector(641, 100), Vector(2.5, 2))
    mover.bounce_edges(640, 240)
    assert mover.velocity == Vector(-2.5, 2)


def test_random_acceleration_is_bounded() -> None:
    rng = random.Random(1)
    lengths = [random_acceleration(rng).mag() for _ in range(1000)]
    assert max(lengths) <= 2
    assert sum(lengths) / len(lengths) == pytest.approx(1, rel=0.1)


def test_toward_points_at_the_target_with_constant_strength() -> None:
    a = toward(Vector(10, 0), Vector(0, 0))
    assert a.xy == pytest.approx((0.2, 0))
    assert toward(Vector(500, 500), Vector(0, 0)).mag() == pytest.approx(0.2)


def test_toward_by_distance_grows_with_distance() -> None:
    near = toward_by_distance(Vector(10, 0), Vector(), 640)
    far = toward_by_distance(Vector(640, 0), Vector(), 640)
    assert near.mag() == pytest.approx(0.2 * 10 / 640)
    assert far.xy == pytest.approx((0.2, 0))


def test_noise_acceleration_is_smooth_and_bounded() -> None:
    accelerate = NoiseAcceleration(Noise(seed=2), strength=0.1)
    values = [accelerate() for _ in range(200)]
    assert all(abs(a.x) <= 0.1 and abs(a.y) <= 0.1 for a in values)
    assert max(a.dist(b) for a, b in itertools.pairwise(values)) < 0.01


def test_train_accelerates_brakes_and_never_reverses() -> None:
    train = Train(Vector(0, 100))
    train.throttle(0.1)
    for _ in range(10):
        train.update(640, 64)
    assert train.velocity.x == pytest.approx(2)
    train.throttle(-0.5)
    for _ in range(10):
        train.update(640, 64)
    assert train.velocity.x == 0
    assert train.acceleration == 0


def test_train_wraps_around() -> None:
    train = Train(Vector(639, 100), Vector(5, 0))
    train.update(640, 64)
    assert train.position.x == -64
