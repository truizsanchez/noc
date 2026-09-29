import itertools
import math

import pytest

from noc.ch03_oscillation.oscillation import (
    DoublePendulum,
    Incline,
    Oscillator,
    Pendulum,
    RotatingMover,
    Spaceship,
    Spinner,
    Spring,
    Wave,
    additive_wave,
    connect,
    noise_wave,
    simple_harmonic,
)
from noc.common.noise import Noise
from noc.common.physics import Mover
from noc.common.vector import Vector


def test_spinner_accelerates_and_damps() -> None:
    spin = Spinner(acceleration=0.01)
    for _ in range(10):
        spin.update()
    assert spin.velocity == pytest.approx(0.1)
    assert spin.angle == pytest.approx(0.55)
    damped = Spinner(velocity=1, damping=0.5)
    damped.update()
    assert damped.velocity == 0.5


def test_rotating_mover_spin_follows_horizontal_acceleration() -> None:
    mover = RotatingMover(Mover(Vector(), mass=1))
    mover.body.apply_force(Vector(0.5, 0))
    mover.update()
    assert mover.spin.velocity == pytest.approx(0.05)
    for _ in range(10):
        mover.body.apply_force(Vector(5, 0))
        mover.update()
    assert mover.spin.velocity == pytest.approx(0.1)  # clamped


def test_spaceship_thrusts_along_its_heading() -> None:
    ship = Spaceship(Mover(Vector(100, 100), top_speed=6))
    ship.thrust()
    ship.update(640, 240, 32)
    assert ship.body.velocity.x == pytest.approx(0)
    assert ship.body.velocity.y < 0  # heading 0 points up
    ship.turn(math.pi / 2)
    for _ in range(200):
        ship.thrust()
        ship.update(640, 240, 32)
    assert ship.body.velocity.mag() <= 6 + 1e-9
    assert -32 <= ship.body.position.x <= 672


def test_simple_harmonic_motion_repeats_every_period() -> None:
    assert simple_harmonic(200, 120, 0) == 0
    assert simple_harmonic(200, 120, 30) == pytest.approx(200)
    assert simple_harmonic(200, 120, 150) == pytest.approx(200)


def test_oscillator_offsets_stay_within_the_amplitude() -> None:
    oscillator = Oscillator(Vector(0.03, 0.05), Vector(100, 50))
    for _ in range(500):
        oscillator.update()
        x, y = oscillator.offset()
        assert abs(x) <= 100
        assert abs(y) <= 50


def test_wave_heights_have_the_given_period() -> None:
    wave = Wave(amplitude=20, period=64, width=160, spacing=8)
    heights = wave.heights()
    assert len(heights) == 20
    assert heights[0] == pytest.approx(0)
    assert heights[2] == pytest.approx(20)  # a quarter period: 16 px, two samples
    assert heights[4] == pytest.approx(0, abs=1e-9)


def test_additive_wave_is_the_sum_of_its_parts() -> None:
    a, b = Wave(10, 100, 0), Wave(5, 50, 0)
    total = additive_wave([a, b], 10)
    dx_a, dx_b = 2 * math.pi / 100 * 8, 2 * math.pi / 50 * 8
    assert total[3] == pytest.approx(10 * math.sin(3 * dx_a) + 5 * math.cos(3 * dx_b))


def test_noise_wave_is_smooth() -> None:
    heights = noise_wave(Noise(seed=1), 0, 50, 0.1)
    assert all(0 <= h <= 1 for h in heights)
    assert max(abs(a - b) for a, b in itertools.pairwise(heights)) < 0.15


def test_spring_follows_hookes_law() -> None:
    spring = Spring(Vector(0, 0), rest_length=100, k=0.2)
    assert spring.force(Vector(0, 150)).xy == pytest.approx((0, -10))  # stretched: pulls back
    assert spring.force(Vector(0, 50)).xy == pytest.approx((0, 10))  # compressed: pushes out
    assert spring.force(Vector(0, 100)).mag() == pytest.approx(0)


def test_spring_length_constraint() -> None:
    spring = Spring(Vector(0, 0), 100)
    bob = Mover(Vector(0, 300), Vector(0, 5))
    spring.constrain(bob, 30, 200)
    assert bob.position == Vector(0, 200)
    assert bob.velocity == Vector()


def test_bob_on_a_damped_spring_settles_where_gravity_balances_the_spring() -> None:
    spring = Spring(Vector(0, 0), 100, k=0.2)
    bob = Mover(Vector(0, 100), mass=24)
    for _ in range(3000):
        bob.apply_force(Vector(0, 2))
        bob.apply_force(spring.force(bob.position))
        bob.velocity *= 0.98
        bob.update()
    assert bob.position.y == pytest.approx(110, abs=0.01)  # stretch = 2 / 0.2


def test_connect_pulls_both_ends_equally() -> None:
    a, b = Mover(Vector(0, 0)), Mover(Vector(0, 50))
    connect(a, b, rest_length=30)
    assert (a.acceleration + b.acceleration).mag() == pytest.approx(0)
    assert b.acceleration.y < 0


def test_pendulum_swings_symmetrically_and_damps() -> None:
    pendulum = Pendulum(Vector(320, 0), 175, damping=1.0)
    angles = []
    for _ in range(400):
        pendulum.update()
        angles.append(pendulum.angle)
    assert min(angles) == pytest.approx(-math.pi / 4, abs=0.01)
    damped = Pendulum(Vector(320, 0), 175)
    for _ in range(2000):
        damped.update()
    assert abs(damped.angle) < 0.01


def test_pendulum_bob_and_dragging() -> None:
    pendulum = Pendulum(Vector(0, 0), 100, angle=0)
    assert pendulum.bob.xy == pytest.approx((0, 100))
    pendulum.point_at(Vector(100, 0))
    assert pendulum.angle == pytest.approx(math.pi / 2)
    assert pendulum.bob.xy == pytest.approx((100, 0))


def energy_range(substeps: int) -> float:
    pendulum = DoublePendulum()
    energies = []
    for _ in range(600):
        pendulum.update(substeps)
        energies.append(pendulum.energy())
    return max(energies) - min(energies)


def test_double_pendulum_energy_drift_shrinks_with_substeps() -> None:
    # The energy scale is m * g * r = 1000.
    assert energy_range(1) > 1000
    assert energy_range(50) < 150


def test_double_pendulum_hangs_still_at_rest() -> None:
    pendulum = DoublePendulum(a1=0, a2=0)
    pendulum.update()
    assert (pendulum.a1, pendulum.a2) == (0, 0)


def test_incline_static_friction_and_sliding() -> None:
    held = Incline(math.radians(10), mu=0.5)  # tan(10°) = 0.18 < 0.5
    held.update()
    assert held.distance == 0
    sliding = Incline(math.radians(30), mu=0.2)
    for _ in range(10):
        sliding.update()
    expected = 0.1 * (math.sin(math.radians(30)) - 0.2 * math.cos(math.radians(30)))
    assert sliding.speed == pytest.approx(10 * expected)
