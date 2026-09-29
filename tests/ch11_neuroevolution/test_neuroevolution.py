import math
import random

import numpy as np
import pytest

from noc.ch11_neuroevolution.games import (
    Bird,
    Box,
    Course,
    Ecosystem,
    Flock,
    Pipe,
    RocketBrains,
    Seekers,
    component_force,
    sense,
    sensor_offsets,
    steering_force,
)
from noc.common.vector import Vector


def test_bird_physics_follow_the_book() -> None:
    bird = Bird()
    bird.update()
    assert bird.y == pytest.approx(120.5)
    assert bird.velocity == pytest.approx(0.5 * 0.95)
    bird.flap()
    assert bird.velocity == pytest.approx(0.475 - 10)
    for _ in range(200):
        bird.update()
    assert bird.y == 240  # resting on the ground


def test_pipe_collision_and_score() -> None:
    pipe = Pipe(top=50, x=40)
    assert pipe.collides(50, 30)  # above the gap
    assert not pipe.collides(50, 100)  # in the gap
    assert not pipe.collides(80, 30)  # beside the pipe
    course = Course(random.Random(1), [Pipe(top=50, x=30)])
    for _ in range(10):
        course.update(bird_x=50)
    assert course.score == 1


def test_course_adds_a_pipe_every_100_frames() -> None:
    course = Course(random.Random(2))
    for _ in range(300):
        course.update()
    assert len(course.pipes) == 4  # at x = 40, 240, 440 and 640


def test_flappy_birds_learn() -> None:
    flock = Flock(size=200, rng=np.random.default_rng(3), course=Course(random.Random(3)))
    first = None
    while flock.generation < 30 and flock.last_best < 2000:
        before = flock.generation
        flock.step()
        if flock.generation > before and first is None:
            first = flock.last_best
    assert first is not None
    assert flock.last_best > 3 * first  # a bird that lives through many more pipes


def test_force_encodings() -> None:
    outputs = np.array([[0.25, 1.0], [0.5, 0.5]])
    force = steering_force(outputs, 2.0)
    assert force[0] == pytest.approx([0, 2])  # a quarter turn, full strength (y down)
    assert force[1] == pytest.approx([-1, 0])
    assert component_force(np.array([[1.0, 0.5]]), 2.0)[0] == pytest.approx([2, 0])


def test_rockets_stop_on_obstacles_and_evolve() -> None:
    rockets = RocketBrains(
        Box(308, 24, 24, 24), [Box(0, 0, 640, 200)], size=20, rng=np.random.default_rng(4)
    )
    for _ in range(rockets.lifespan):
        rockets.step()
    assert rockets.hit_obstacle.any()
    assert (rockets.fitness() >= 0).all()
    rockets.step()  # the next generation
    assert rockets.generation == 1
    assert (rockets.position == [320, 220]).all()


def test_rockets_can_sense_velocity() -> None:
    rockets = RocketBrains(Box(308, 24, 24, 24), [], size=5, use_velocity=True)
    assert rockets.inputs().shape == (5, 4)


def test_seekers_collect_fitness_on_the_glow() -> None:
    seekers = Seekers(size=20, rng=np.random.default_rng(5), components=True)
    seekers.position[:] = seekers.glow.position
    seekers.step()
    assert seekers.fitness.sum() > 0


def test_sensors() -> None:
    offsets = sensor_offsets(4, 10)
    assert offsets[1] == pytest.approx([0, 10])
    values = sense(Vector(0, 0), offsets, Vector(10, 0), 5)
    assert values[0] == pytest.approx(1)  # the right whisker is at the food's center
    assert values[2] == 0


def test_ecosystem_keeps_a_minimum_population() -> None:
    eco = Ecosystem(rng=np.random.default_rng(6))
    for _ in range(3000):
        eco.step()
        assert len(eco.brains) == len(eco.position) == len(eco.health) == len(eco)
    assert len(eco) >= eco.minimum - 1  # the minimum is restored at the start of each step
    assert math.isfinite(float(eco.position.sum()))
