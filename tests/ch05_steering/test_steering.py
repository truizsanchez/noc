import math
import random

import numpy as np
import pytest

from noc.ch05_steering.flocking import BinLattice, Flock, QuadTree, Rect, SinCosTable
from noc.ch05_steering.vehicle import (
    FlockWeights,
    FlowField,
    Path,
    Vehicle,
    Wander,
    align,
    arrive,
    closest_on_segment,
    cohere,
    evade,
    flee,
    flock,
    follow_field,
    follow_path,
    normal_point,
    pursue,
    seek,
    separate,
    stay_within,
)
from noc.common.noise import Noise
from noc.common.vector import Vector


def test_vehicle_top_speed_is_its_max_speed() -> None:
    v = Vehicle(Vector(), max_speed=3)
    v.apply_force(Vector(100, 0))
    v.update()
    assert v.velocity.mag() == pytest.approx(3)


def test_seek_is_desired_minus_velocity_limited() -> None:
    v = Vehicle(Vector(), Vector(0, 1), max_speed=4, max_force=10)
    assert seek(v, Vector(10, 0)).xy == pytest.approx((4, -1))
    v.max_force = 0.1
    assert seek(v, Vector(10, 0)).mag() == pytest.approx(0.1)
    assert flee(v, Vector(10, 0)).x < 0


def test_seeking_vehicle_reaches_its_target() -> None:
    v = Vehicle(Vector(0, 0), max_speed=4, max_force=0.2)
    for _ in range(300):
        v.apply_force(arrive(v, Vector(300, 100)))
        v.update()
    assert v.position.dist(Vector(300, 100)) < 1
    assert v.velocity.mag() < 0.1  # arrive slows down


def test_arrive_desires_less_speed_near_the_target() -> None:
    v = Vehicle(Vector(), max_speed=4, max_force=100)
    assert arrive(v, Vector(50, 0)).x == pytest.approx(2)  # half the slowing radius
    assert arrive(v, Vector(500, 0)).x == pytest.approx(4)


def test_pursue_aims_ahead_and_evade_away() -> None:
    hunter = Vehicle(Vector(0, 0), max_force=100)
    quarry = Vehicle(Vector(100, 0), Vector(0, 5))
    force = pursue(hunter, quarry, lookahead=10)  # aims at (100, 50)
    assert force.heading() == pytest.approx(math.atan2(50, 100))
    assert evade(quarry, hunter).x > 0


def test_wander_target_is_on_a_circle_ahead() -> None:
    v = Vehicle(Vector(100, 100), Vector(1, 0))
    wander = Wander(rng=random.Random(1))
    center, target = wander.target(v)
    assert center == Vector(180, 100)
    assert target.dist(center) == pytest.approx(25)


def test_stay_within_turns_back_near_a_wall() -> None:
    v = Vehicle(Vector(10, 100), Vector(-3, 1), max_speed=3, max_force=100)
    force = stay_within(v, 640, 240, 25)
    assert (v.velocity + force).x > 0
    assert stay_within(Vehicle(Vector(320, 120)), 640, 240, 25) == Vector()


def test_flow_field_lookup_clamps_to_the_grid() -> None:
    field = FlowField(20, np.zeros((32, 12)))
    field.angles[31, 0] = math.pi / 2
    assert field.lookup(Vector(1000, -50)).xy == pytest.approx((0, 1))
    noisy = FlowField.from_noise(640, 240, 20, Noise(seed=1))
    assert (noisy.cols, noisy.rows) == (32, 12)


def test_swirl_field_circles_the_center() -> None:
    field = FlowField.swirl(640, 240, 20)
    right = field.lookup(Vector(630, 120))
    assert right.xy == pytest.approx((0, 1), abs=0.1)  # heading down on the right: clockwise


def test_following_a_field_matches_its_direction() -> None:
    field = FlowField(20, np.full((32, 12), math.pi / 2))
    v = Vehicle(Vector(100, 100), max_speed=2, max_force=0.5)
    for _ in range(20):
        v.apply_force(follow_field(v, field))
        v.update()
    assert v.velocity.xy == pytest.approx((0, 2))


def test_normal_point_and_segment_clamp() -> None:
    a, b = Vector(0, 0), Vector(10, 0)
    assert normal_point(Vector(5, 7), a, b) == Vector(5, 0)
    assert normal_point(Vector(15, 7), a, b) == Vector(15, 0)
    assert closest_on_segment(Vector(15, 7), a, b) == Vector(10, 0)
    # Works for segments going right to left, unlike the book's x test.
    assert closest_on_segment(Vector(-5, 3), b, a) == Vector(0, 0)


def test_path_following_only_steers_outside_the_radius() -> None:
    path = Path([Vector(0, 100), Vector(640, 100)], radius=20)
    inside = Vehicle(Vector(100, 105), Vector(2, 0))
    assert follow_path(inside, path).force == Vector()
    outside = Vehicle(Vector(100, 180), Vector(2, 0), max_force=100)
    info = follow_path(outside, path)
    assert info.normal == Vector(150, 100)
    assert info.target == Vector(160, 100)
    assert info.force.y < 0


def test_vehicles_converge_onto_a_closed_path() -> None:
    path = Path([Vector(50, 50), Vector(590, 50), Vector(590, 190), Vector(50, 190)], closed=True)
    v = Vehicle(Vector(320, 120), Vector(1, 0), max_speed=3, max_force=0.3)
    for _ in range(600):
        v.apply_force(follow_path(v, path, 25, 25).force)
        v.update()
    distance = min(
        v.position.dist(closest_on_segment(v.position, a, b)) for a, b in path.segments()
    )
    assert distance < path.radius + 5


def test_separation_pushes_apart_and_ignores_the_far() -> None:
    a, b = Vehicle(Vector(0, 0), max_force=100), Vehicle(Vector(5, 0))
    far = Vehicle(Vector(500, 0))
    assert separate(a, [a, b, far], 24).x < 0
    assert separate(a, [a, far], 24) == Vector()


def test_align_and_cohere() -> None:
    a = Vehicle(Vector(0, 0), max_force=100)
    b = Vehicle(Vector(10, 0), Vector(0, 2))
    assert align(a, [a, b]).xy == pytest.approx((0, 4))
    assert cohere(a, [a, b]).xy == pytest.approx((4, 0))


def make_boids(rng: random.Random, n: int) -> list[Vehicle]:
    return [
        Vehicle(
            Vector(rng.uniform(0, 200), rng.uniform(0, 100)),
            Vector(rng.uniform(-1, 1), rng.uniform(-1, 1)),
            max_speed=3,
            max_force=0.05,
            r=3,
        )
        for _ in range(n)
    ]


def test_single_pass_flock_matches_the_three_behaviors() -> None:
    boids = make_boids(random.Random(2), 30)
    w = FlockWeights()
    for boid in boids:
        expected = (
            separate(boid, boids, w.separation_distance) * w.separation
            + align(boid, boids, w.neighbor_distance) * w.alignment
            + cohere(boid, boids, w.neighbor_distance) * w.cohesion
        )
        assert flock(boid, boids, w).xy == pytest.approx(expected.xy)


def test_numpy_flock_matches_the_object_flock() -> None:
    boids = make_boids(random.Random(3), 40)
    arrays = Flock(
        np.array([b.position.xy for b in boids]), np.array([b.velocity.xy for b in boids])
    )
    forces = arrays.forces()
    for boid, force in zip(boids, forces, strict=True):
        assert tuple(force) == pytest.approx(flock(boid, boids, FlockWeights()).xy, abs=1e-12)


def test_numpy_flock_update_limits_speed_and_wraps() -> None:
    flock_ = Flock(np.array([[645.0, 10.0]]), np.array([[10.0, 0.0]]))
    flock_.update(np.zeros((1, 2)), 640, 240)
    assert np.linalg.norm(flock_.velocities[0]) == pytest.approx(3)
    assert flock_.positions[0, 0] == -3


def test_bin_lattice_finds_the_same_neighbors_as_brute_force() -> None:
    boids = make_boids(random.Random(4), 100)
    grid = BinLattice(50)
    grid.rebuild(boids)
    for boid in boids:
        near = {id(b) for b in grid.neighbors(boid)}
        brute = {id(b) for b in boids if b.position.dist(boid.position) < 50}
        assert brute <= near


def test_quadtree_query_matches_a_scan() -> None:
    boids = make_boids(random.Random(5), 200)
    tree = QuadTree(Rect(100, 50, 110, 60), capacity=4)
    assert all(tree.insert(b) for b in boids)
    assert tree.children  # it subdivided
    area = Rect(60, 40, 25, 15)
    found = {id(b) for b in tree.query(area)}
    scan = {id(b) for b in boids if area.contains(*b.position.xy)}
    assert found == scan
    assert not tree.insert(Vehicle(Vector(1000, 1000)))


def test_sin_cos_table() -> None:
    table = SinCosTable(0.5)
    assert table.period == 720
    assert table.sin[table.index(90)] == pytest.approx(1)
    assert table.cos[table.index(360 + 60)] == pytest.approx(0.5)
