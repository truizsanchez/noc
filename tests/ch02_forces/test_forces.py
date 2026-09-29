import random

import numpy as np
import pytest

from noc.ch02_forces.forces import (
    Balloon,
    Galaxy,
    attract_all,
    attract_and_repel,
    edge_repulsion,
    fan,
    figure_eight,
)
from noc.common.noise import Noise
from noc.common.physics import Mover
from noc.common.vector import Vector


def test_balloon_rises_and_bounces_off_the_top() -> None:
    balloon = Balloon(Mover(Vector(320, 120), top_speed=5), noise=Noise(seed=1))
    ys = []
    for _ in range(600):
        balloon.step(640, 48)
        ys.append(balloon.body.position.y)
    assert min(ys) == pytest.approx(24)
    assert ys[100] < 120


def test_edge_repulsion_is_stronger_closer_to_the_edge() -> None:
    assert edge_repulsion(Vector(320, 120), 640, 240) == Vector()
    near = edge_repulsion(Vector(5, 120), 640, 240)
    farther = edge_repulsion(Vector(30, 120), 640, 240)
    assert near.x > farther.x > 0
    assert edge_repulsion(Vector(320, 235), 640, 240).y < 0


def test_fan_blows_away_from_the_source_and_fades() -> None:
    assert fan(Vector(), Vector(100, 0)).xy == pytest.approx((0.5 * (1 - 100 / 300), 0))
    assert fan(Vector(), Vector(400, 0)) == Vector()


def test_attract_and_repel_balance_at_the_rest_distance() -> None:
    far = attract_and_repel(Vector(), 10, Vector(100, 0), 1, rest_distance=60)
    close = attract_and_repel(Vector(), 10, Vector(30, 0), 1, rest_distance=60)
    rest = attract_and_repel(Vector(), 10, Vector(60, 0), 1, rest_distance=60)
    assert far.x < 0  # pulled toward the source
    assert close.x > 0  # pushed away
    assert rest.mag() == pytest.approx(0)


def test_attract_all_conserves_momentum() -> None:
    rng = random.Random(2)
    bodies = [
        Mover(Vector(rng.uniform(0, 640), rng.uniform(0, 240)), mass=rng.uniform(0.1, 2))
        for _ in range(5)
    ]

    def momentum() -> Vector:
        total = Vector()
        for b in bodies:
            total += b.velocity * b.mass
        return total

    for _ in range(100):
        attract_all(bodies, distance_range=(5, 1e9))
        for b in bodies:
            b.update()
    assert momentum().mag() == pytest.approx(0, abs=1e-9)


def test_figure_eight_returns_to_its_start_after_one_period() -> None:
    bodies = figure_eight(Vector(320, 120), 150, 600)
    start = [b.position for b in bodies]
    for _ in range(600):
        attract_all(bodies, distance_range=(1, float("inf")))
        for b in bodies:
            b.update()
    for body, origin in zip(bodies, start, strict=True):
        assert body.position.dist(origin) < 5


def test_galaxy_stars_orbit_the_sun() -> None:
    galaxy = Galaxy.spiral(random.Random(3), count=30)
    start = np.linalg.norm(galaxy.positions, axis=1)
    for _ in range(50):
        galaxy.step()
    radii = np.linalg.norm(galaxy.positions, axis=1)
    assert np.isfinite(radii).all()
    assert np.median(radii) < 3 * np.median(start)
