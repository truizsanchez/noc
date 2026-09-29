import itertools
import math

import numpy as np
import pytest

from noc.common.noise import SIZE, Noise, lcg_values, scaled_cosine


def test_lcg_matches_numerical_recipes() -> None:
    # z1 = (1664525 * 0 + 1013904223) mod 2^32, z2 = (1664525 * z1 + 1013904223) mod 2^32.
    first, second = lcg_values(0, 2)
    assert first == 1013904223 / 2**32
    assert second == ((1664525 * 1013904223 + 1013904223) % 2**32) / 2**32


def test_scaled_cosine_eases_from_zero_to_one() -> None:
    assert scaled_cosine(0) == 0
    assert scaled_cosine(0.5) == pytest.approx(0.5)
    assert scaled_cosine(1) == 1


def test_seeded_noise_is_deterministic() -> None:
    a, b = Noise(seed=42), Noise(seed=42)
    points = [(0.1, 0, 0), (3.7, 1.2, 0), (100.5, 7.25, 3.5)]
    assert [a(*p) for p in points] == [b(*p) for p in points]
    assert Noise(seed=1)(0.5) != a(0.5)


def test_single_octave_at_a_lattice_point_is_half_the_table_value() -> None:
    n = Noise(seed=7, octaves=1)
    table = lcg_values(7, SIZE + 1)
    # At x = 3 (y = z = 0) no blending happens: the result is the table entry times 0.5.
    assert n(3) == pytest.approx(0.5 * table[3])
    # y and z select rows 16 and 256 entries apart.
    assert n(0, 1) == pytest.approx(0.5 * table[16])
    assert n(0, 0, 1) == pytest.approx(0.5 * table[256])


def test_values_stay_between_zero_and_one() -> None:
    n = Noise(seed=3)
    values = [n(x * 0.37, y * 0.11) for x, y in itertools.product(range(60), range(60))]
    assert min(values) >= 0
    assert max(values) < 1


def test_noise_is_smooth_but_random_is_not() -> None:
    n = Noise(seed=5)
    steps = [abs(n(t * 0.01 + 0.01) - n(t * 0.01)) for t in range(500)]
    assert max(steps) < 0.03


def test_negative_coordinates_mirror_positive_ones() -> None:
    n = Noise(seed=9)
    assert n(-2.5, -1.25) == n(2.5, 1.25)


def test_detail_changes_octaves_and_ignores_non_positive_values() -> None:
    n = Noise(seed=11)
    n.detail(8, 0.25)
    assert (n.octaves, n.falloff) == (8, 0.25)
    n.detail(0, -1)
    assert (n.octaves, n.falloff) == (8, 0.25)


def test_more_falloff_means_rougher_noise() -> None:
    smooth, rough = Noise(seed=13, falloff=0.2), Noise(seed=13, falloff=0.8)

    def roughness(n: Noise) -> float:
        return sum(abs(n(t * 0.01 + 0.01) - n(t * 0.01)) for t in range(1000))

    assert roughness(rough) > roughness(smooth)


@pytest.mark.parametrize(("octaves", "z"), [(1, 0.0), (4, 0.0), (6, 2.75)])
def test_grid_matches_scalar_noise(octaves: int, z: float) -> None:
    n = Noise(seed=17, octaves=octaves)
    xs, ys = np.meshgrid(np.arange(0, 3, 0.07), np.arange(0, 1.5, 0.05))
    grid = n.grid(xs, ys, z)
    scalar = np.vectorize(lambda x, y: n(x, y, z))(xs, ys)
    np.testing.assert_allclose(grid, scalar, rtol=0, atol=1e-12)
    assert not math.isnan(float(grid.sum()))
