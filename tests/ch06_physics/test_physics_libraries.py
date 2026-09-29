import math

import numpy as np
import pymunk
import pytest

from noc.ch06_physics import integration, world
from noc.ch06_physics.soft import PymunkModel, VerletModel, cloth, soft_body, string
from noc.ch06_physics.verlet import Attraction, VerletPhysics
from noc.common.vector import Vector


# --- our Verlet engine ---------------------------------------------------------------------
def test_verlet_keeps_velocity_as_the_last_step() -> None:
    physics = VerletPhysics()
    i = physics.add_particle(0, 0)
    physics.previous[i] = (-2, 0)  # moving 2 px per frame to the right
    physics.update()
    physics.update()
    assert tuple(physics.positions[i]) == pytest.approx((4, 0))


def test_gravity_accelerates_like_euler_steps() -> None:
    physics = VerletPhysics(gravity=(0, 0.5))
    i = physics.add_particle(0, 0)
    for _ in range(3):
        physics.update()
    assert physics.positions[i][1] == pytest.approx(0.5 + 1.0 + 1.5)


def test_locked_particles_and_move_to() -> None:
    physics = VerletPhysics(gravity=(0, 1))
    i = physics.add_particle(5, 5)
    physics.lock(i)
    physics.update()
    assert tuple(physics.positions[i]) == (5, 5)
    physics.lock(i, False)
    physics.move_to(i, 50, 60)
    assert tuple(physics.previous[i]) == (50, 60)  # no velocity from the jump


def test_spring_relaxation_reaches_the_rest_length() -> None:
    physics = VerletPhysics()
    a, b = physics.add_particle(0, 0), physics.add_particle(200, 0)
    physics.add_spring(a, b, 100, strength=0.5)
    physics.lock(a)
    physics.relax_springs()
    # Toxiclibs splits the correction between both ends; the locked end ignores its half.
    assert physics.positions[b][0] == pytest.approx(200 - 0.5 * 100 / 2)
    for _ in range(40):
        physics.relax_springs()
    assert physics.spring_lengths()[0] == pytest.approx(100, abs=0.01)


def test_default_rest_length_is_the_current_distance() -> None:
    physics = VerletPhysics()
    a, b = physics.add_particle(0, 0), physics.add_particle(3, 4)
    physics.add_spring(a, b)
    assert physics.rest_lengths[0] == 5


def test_min_distance_springs_only_push() -> None:
    physics = VerletPhysics()
    a, b = physics.add_particle(0, 0), physics.add_particle(200, 0)
    physics.add_spring(a, b, 100, 0.5, min_distance=True)
    physics.relax_springs()
    assert physics.spring_lengths()[0] == pytest.approx(200)
    physics.move_to(b, 50, 0)
    physics.relax_springs()
    assert physics.spring_lengths()[0] > 50


def test_spring_groups_never_share_a_particle() -> None:
    physics = VerletModel(cloth(8, 6)).physics
    groups = physics.spring_groups()
    assert sum(len(g) for g in groups) == len(physics.rest_lengths)
    for group in groups:
        ends = physics.spring_ends[group].ravel()
        assert len(set(ends.tolist())) == len(ends)
    assert len(groups) <= 5  # a grid needs 4 colors


def test_springs_conserve_momentum() -> None:
    physics = VerletPhysics()
    for x, y in [(0, 0), (30, 5), (10, 40), (-20, 10)]:
        physics.add_particle(x, y)
    for a in range(4):
        for b in range(a + 1, 4):
            physics.add_spring(a, b, 25, 0.1)
    before = physics.positions.mean(axis=0)
    for _ in range(20):
        physics.update()
    assert physics.positions.mean(axis=0) == pytest.approx(before)


def test_attraction_behavior() -> None:
    physics = VerletPhysics()
    center = physics.add_particle(0, 0)
    near = physics.add_particle(10, 0)
    far = physics.add_particle(100, 0)
    physics.attractions.append(Attraction(center, 20, 1.0))
    forces = physics.forces()
    assert tuple(forces[near]) == pytest.approx((-(1 - 100 / 400), 0))
    assert tuple(forces[far]) == (0, 0)
    assert tuple(forces[center]) == (0, 0)


def test_bounds_clamp_positions() -> None:
    physics = VerletPhysics(gravity=(0, 50), bounds=(0, 0, 640, 240))
    i = physics.add_particle(10, 230)
    physics.update()
    assert physics.positions[i][1] == 240


# --- soft bodies, both engines -------------------------------------------------------------
@pytest.mark.parametrize("model", [VerletModel, PymunkModel])
@pytest.mark.parametrize(
    "topology",
    [string(640, 240), soft_body(640, 240), cloth(10, 6)],
    ids=["string", "body", "cloth"],
)
def test_soft_bodies_stay_stable(model: type, topology: object) -> None:
    m = model(topology)
    for _ in range(200):
        m.step()
    points = np.array(m.points())
    assert np.isfinite(points).all()
    lengths = [math.dist(points[s.a], points[s.b]) / s.rest_length for s in m.topology.springs]
    assert min(lengths) > 0.5 and max(lengths) < 2.0


@pytest.mark.parametrize("model", [VerletModel, PymunkModel])
def test_pinned_particles_stay_put(model: type) -> None:
    topology = string(640, 240)
    m = model(topology)
    for _ in range(50):
        m.step()
    assert m.points()[0] == pytest.approx(topology.points[0])


@pytest.mark.parametrize("model", [VerletModel, PymunkModel])
def test_dragging_moves_a_particle(model: type) -> None:
    m = model(soft_body(640, 240))
    m.drag(0, 100, 100)
    assert m.points()[0] == pytest.approx((100, 100))


# --- pymunk helpers ------------------------------------------------------------------------
def test_matter_like_bodies_fall_and_land() -> None:
    space = world.new_space()
    world.add_box(space, 320, 235, 640, 10, static=True)
    box = world.add_box(space, 320, 100, 20, 20)
    assert box.mass == pytest.approx(20 * 20 * world.DENSITY)
    for _ in range(300):
        world.step(space)
    assert box.position.y == pytest.approx(220, abs=1)


def test_lollipop_is_one_body_with_two_shapes() -> None:
    space = world.new_space()
    body = world.add_lollipop(space, 100, 100)
    kinds = sorted(type(shape).__name__ for shape in body.shapes)
    assert kinds == ["Circle", "Poly"]


def test_polygon_and_removal() -> None:
    space = world.new_space()
    body = world.add_polygon(space, 0, 0, [(-10, -10), (20, -15), (15, 0), (0, 10), (-20, 15)])
    assert len(space.bodies) == 1
    world.remove(space, body)
    assert not space.bodies


def test_mouse_grab_drags_a_body() -> None:
    space = world.new_space(gravity=0)
    box = world.add_box(space, 100, 100, 40, 40)
    grab = world.MouseGrab(space, max_force=1000)
    assert grab.press(105, 100)
    for _ in range(60):
        grab.move(200, 100)
        world.step(space)
    assert box.position.x > 150
    grab.release()
    assert not grab.press(500, 500)  # nothing there


def test_pymunk_pin_joint_keeps_the_pendulum_length() -> None:
    space = world.new_space()
    anchor = world.add_circle(space, 320, 10, 12, static=True)
    bob = world.add_circle(space, 420, -90, 12)
    space.add(pymunk.PinJoint(anchor, bob))
    length = (bob.position - anchor.position).length
    for _ in range(200):
        world.step(space)
    assert (bob.position - anchor.position).length == pytest.approx(length, rel=0.01)


# --- integration methods -------------------------------------------------------------------
def orbit_energies(method: str, steps: int = 1000) -> tuple[float, float]:
    center, gm = Vector(0, 0), 440.0
    gravity = integration.gravity_toward(center, gm)
    start, velocity = Vector(100, 0), Vector(0, 2.1)
    if method == "verlet":
        vb = integration.VerletBody.launched(start, velocity)
        first = integration.orbital_energy(vb.position, velocity, center, gm)
        for _ in range(steps):
            integration.verlet(vb, gravity)
        return first, integration.orbital_energy(vb.position, vb.velocity, center, gm)
    body = integration.Body(start, velocity)
    step = integration.explicit_euler if method == "euler" else integration.semi_implicit_euler
    first = integration.orbital_energy(body.position, body.velocity, center, gm)
    for _ in range(steps):
        step(body, gravity)
    return first, integration.orbital_energy(body.position, body.velocity, center, gm)


def test_explicit_euler_gains_energy_on_an_orbit() -> None:
    first, last = orbit_energies("euler")
    assert last > first + 0.5  # the orbit spirals outward


@pytest.mark.parametrize("method", ["semi-implicit", "verlet"])
def test_symplectic_methods_keep_the_orbit(method: str) -> None:
    first, last = orbit_energies(method)
    assert last == pytest.approx(first, abs=0.2)
