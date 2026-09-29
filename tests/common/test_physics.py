import pytest

from noc.common.physics import Liquid, Mover, attraction, drag, friction, weight
from noc.common.vector import Vector


def test_force_accumulation_and_mass() -> None:
    mover = Mover(Vector(), mass=2)
    mover.apply_force(Vector(0, 1))
    mover.apply_force(Vector(1, 0))
    assert mover.acceleration == Vector(0.5, 0.5)
    mover.update()
    assert mover.velocity == Vector(0.5, 0.5)
    assert mover.position == Vector(0.5, 0.5)
    assert mover.acceleration == Vector()  # cleared every frame


def test_heavier_objects_accelerate_less_under_the_same_force() -> None:
    light, heavy = Mover(Vector(), mass=2), Mover(Vector(), mass=10)
    for mover in (light, heavy):
        mover.apply_force(Vector(0.1, 0))
        mover.update()
    assert light.velocity.x == pytest.approx(5 * heavy.velocity.x)


def test_weight_makes_every_mass_fall_alike() -> None:
    light, heavy = Mover(Vector(), mass=2), Mover(Vector(), mass=10)
    for mover in (light, heavy):
        mover.apply_force(weight(mover.mass))
        mover.update()
    assert light.velocity == heavy.velocity == Vector(0, 0.1)


def test_bounce_keeps_the_circle_inside_and_loses_energy() -> None:
    mover = Mover(Vector(700, 250), Vector(3, 4))
    mover.bounce(640, 240, radius=10, restitution=0.9)
    assert mover.position == Vector(630, 230)
    assert mover.velocity.xy == pytest.approx((-2.7, -3.6))
    ceiling = Mover(Vector(50, -5), Vector(0, -2))
    ceiling.bounce(640, 240, top=True)
    assert ceiling.position == Vector(50, 0)
    assert ceiling.velocity == Vector(0, 2)
    open_top = Mover(Vector(50, -5), Vector(0, -2))
    open_top.bounce(640, 240)
    assert open_top.position == Vector(50, -5)


def test_friction_opposes_motion_with_constant_magnitude() -> None:
    f = friction(Vector(3, 4), 0.1)
    assert f.xy == pytest.approx((-0.06, -0.08))
    assert friction(Vector(), 0.1) == Vector()


def test_drag_grows_with_the_square_of_speed() -> None:
    slow, fast = drag(Vector(1, 0), 0.1), drag(Vector(2, 0), 0.1)
    assert slow.xy == pytest.approx((-0.1, 0))
    assert fast.xy == pytest.approx((-0.4, 0))


def test_limited_drag_can_stop_but_not_reverse() -> None:
    mover = Mover(Vector(), Vector(5, 0), mass=1)
    assert drag(mover.velocity, 2).mag() == pytest.approx(50)  # would send it back at 45
    mover.apply_force(drag(mover.velocity, 2, mass=mover.mass))
    mover.update()
    assert mover.velocity.x == pytest.approx(0)


def test_attraction_is_inverse_square_within_the_clamped_range() -> None:
    near = attraction(Vector(10, 0), 20, Vector(), 2, distance_range=(5, 100))
    far = attraction(Vector(20, 0), 20, Vector(), 2, distance_range=(5, 100))
    assert near.xy == pytest.approx((40 / 100, 0))
    assert far.x == pytest.approx(near.x / 4)
    clamped = attraction(Vector(100, 0), 20, Vector(), 2)  # default range 5-25
    assert clamped.x == pytest.approx(40 / 625)


def test_liquid_contains() -> None:
    liquid = Liquid(0, 120, 640, 120, 0.1)
    assert liquid.contains(Vector(10, 200))
    assert not liquid.contains(Vector(10, 100))
