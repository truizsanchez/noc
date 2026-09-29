import itertools
import math
import random

import numpy as np
import pytest

from noc.ch00_randomness import walkers
from noc.ch00_randomness.distributions import Histogram, Splatter, accept_reject, normal_pdf
from noc.ch00_randomness.landscapes import (
    Terrain,
    grayscale,
    hue_and_brightness,
    noise_field,
    project,
)
from noc.ch00_randomness.walkers import NoiseWalker, Walker
from noc.common.noise import Noise


def test_four_directions_are_equally_likely() -> None:
    rng = random.Random(1)
    walker = Walker(0, 0, walkers.four_directions)
    steps = [walkers.four_directions(rng, walker) for _ in range(8000)]
    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        assert steps.count(direction) == pytest.approx(2000, rel=0.1)


def test_tends_right_moves_right_40_percent_of_the_time() -> None:
    rng = random.Random(2)
    walker = Walker(0, 0, walkers.tends_right)
    steps = [walkers.tends_right(rng, walker) for _ in range(10_000)]
    assert steps.count((1, 0)) / len(steps) == pytest.approx(0.4, abs=0.02)


def test_walker_stays_inside_its_bounds() -> None:
    walker = Walker(0, 0, lambda rng, w: (-5, 50), bounds=(640, 240))
    walker.step()
    assert (walker.x, walker.y) == (0, 50)
    for _ in range(10):
        walker.step()
    assert walker.y == 239


def test_skewed_walk_drifts_down_and_right() -> None:
    walker = Walker(0, 0, walkers.skewed, rng=random.Random(3))
    for _ in range(10_000):
        walker.step()
    # The mean step is 0.125 on each axis.
    assert walker.x == pytest.approx(1250, rel=0.2)
    assert walker.y == pytest.approx(1250, rel=0.2)


def test_toward_moves_to_the_target_more_often_than_away() -> None:
    walker = Walker(0, 0, walkers.toward(lambda: (1000, 1000)), rng=random.Random(4))
    for _ in range(2000):
        walker.step()
    # Half the steps go toward the target, the other half are unbiased.
    assert walker.x + walker.y == pytest.approx(1000, rel=0.15)


def test_gaussian_steps_have_the_requested_spread() -> None:
    rng = random.Random(5)
    walker = Walker(0, 0, walkers.gaussian(3))
    dxs = [walkers.gaussian(3)(rng, walker)[0] for _ in range(10_000)]
    mean = sum(dxs) / len(dxs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in dxs) / len(dxs))
    assert mean == pytest.approx(0, abs=0.1)
    assert sd == pytest.approx(3, rel=0.05)


def test_quadratic_steps_favor_long_ones() -> None:
    rng = random.Random(6)
    rule = walkers.quadratic(5)
    walker = Walker(0, 0, rule)
    lengths = [abs(rule(rng, walker)[0]) for _ in range(5000)]
    assert max(lengths) <= 5
    # With density 3x^2 on [0, 1), the mean is 3/4 of the maximum.
    assert sum(lengths) / len(lengths) == pytest.approx(3.75, rel=0.05)


def test_noise_walker_uses_two_independent_parts_of_the_noise() -> None:
    walker = NoiseWalker(640, 240, Noise(seed=7))
    positions = []
    for _ in range(300):
        walker.step()
        positions.append((walker.x, walker.y))
    assert all(0 <= x <= 640 and 0 <= y <= 240 for x, y in positions)
    assert any(abs(x / 640 - y / 240) > 0.05 for x, y in positions)
    jumps = [math.dist(a, b) for a, b in itertools.pairwise(positions)]
    assert max(jumps) < 20  # smooth motion


def test_noise_steps_stay_within_one_pixel() -> None:
    rule = walkers.noise_steps(Noise(seed=8))
    walker = Walker(0, 0, rule)
    for _ in range(100):
        dx, dy = rule(random.Random(), walker)
        assert -1 <= dx <= 1
        assert -1 <= dy <= 1


def test_histogram_bins() -> None:
    histogram = Histogram(4)
    for value in (0.0, 0.24, 0.25, 0.99, 1.0):
        histogram.add(value)
    assert histogram.counts == [2, 1, 0, 2]


def test_accept_reject_follows_the_probability_curve() -> None:
    rng = random.Random(9)
    histogram = Histogram(10)
    for _ in range(20_000):
        histogram.add(accept_reject(lambda x: x, rng))
    # Density 2x: bin i gets (2i + 1) / 100 of the samples.
    for i, count in enumerate(histogram.counts):
        assert count / 20_000 == pytest.approx((2 * i + 1) / 100, abs=0.01)


def test_normal_pdf() -> None:
    assert normal_pdf(0) == pytest.approx(1 / math.sqrt(2 * math.pi))
    assert normal_pdf(1, 1, 2) == pytest.approx(normal_pdf(0) / 2)
    assert normal_pdf(-1.5) == normal_pdf(1.5)


def test_splatter_dots_cluster_around_the_center() -> None:
    rng = random.Random(10)
    dots = [Splatter().dot(rng, 240) for _ in range(2000)]
    within = sum(1 for x, y, _, _ in dots if abs(x) < 0.25 and abs(y) < 0.25)
    assert within / len(dots) == pytest.approx(0.68**2, abs=0.05)
    assert all(size > 0 and all(0 <= c <= 255 for c in rgb) for _, _, size, rgb in dots)


def test_noise_images() -> None:
    values = noise_field(Noise(seed=11), 32, 16, (0.01, 0.01))
    assert values.shape == (16, 32)
    gray = grayscale(values)
    assert gray.shape == (16, 32, 3)
    assert gray.dtype == np.uint8
    color = hue_and_brightness(np.array([[0.0, 1 / 3, 0.5]]))
    assert color[0, 0].tolist() == [0, 0, 0]  # brightness 0
    assert color[0, 1].tolist() == [0, 216, 0]  # hue 120, brightness 85 (1/3 * 255)
    assert color[0, 2].tolist() == [0, 255, 255]  # hue 180, clamped to full brightness


def test_terrain_elevations_and_projection() -> None:
    terrain = Terrain(Noise(seed=12))
    assert terrain.elevations.shape == (40, 20)
    assert (terrain.elevations >= -120).all()
    assert (terrain.elevations <= 120).all()
    before = terrain.elevations.copy()
    terrain.calculate()
    assert not np.array_equal(before, terrain.elevations)
    # The origin sits 20 below the center (the translate) and 200 behind the canvas plane.
    screen = project(np.array([[0.0, 0.0, 0.0]]), 0.0, 640, 240)[0]
    eye = 120 / math.tan(math.pi / 6)
    assert screen[0] == pytest.approx(320)
    assert screen[1] == pytest.approx(120 + 20 * eye / (eye + 200))
    assert screen[2] == pytest.approx(eye + 200)
