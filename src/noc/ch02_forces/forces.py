"""Chapter 2's exercises: forces designed for a purpose, and systems of attracting bodies."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from noc.common.mathutils import remap
from noc.common.noise import Noise
from noc.common.physics import Mover, attraction
from noc.common.vector import Vector

type FloatArray = npt.NDArray[np.float64]


@dataclass
class Balloon:
    """Exercise 2.1: helium lifts it, a Perlin-noise breeze pushes it sideways."""

    body: Mover
    helium: Vector = Vector(0, -0.02)
    noise: Noise = field(default_factory=Noise)
    wind_offset: float = 1000.0

    def step(self, width: float, diameter: float) -> None:
        wind = remap(self.noise(self.wind_offset), 0, 1, -0.01, 0.01)
        self.wind_offset += 0.01
        self.body.apply_force(self.helium)
        self.body.apply_force(Vector(wind, 0))
        self.body.update()
        x, y = self.body.position
        if y < diameter / 2:  # bounce off the top, losing a quarter of the speed
            self.body.position = Vector(x, diameter / 2)
            self.body.velocity = Vector(self.body.velocity.x, -self.body.velocity.y * 0.75)
        if not 0 <= x <= width:
            self.body.position = Vector(x % width, self.body.position.y)


def edge_repulsion(
    position: Vector, width: float, height: float, reach: float = 60, strength: float = 1
) -> Vector:
    """Exercise 2.3: invisible walls that push harder the closer the object gets.

    Each edge within ``reach`` pushes inward with ``strength * (1 - distance / reach)``.
    """

    def push(distance: float) -> float:
        return strength * max(0.0, 1 - distance / reach)

    x, y = position
    return Vector(push(x) - push(width - x), push(y) - push(height - y))


def fan(source: Vector, target: Vector, strength: float = 0.5, reach: float = 300) -> Vector:
    """Exercise 2.5: wind blowing away from ``source``, fading out linearly by ``reach``."""
    direction = target - source
    magnitude = strength * max(0.0, 1 - direction.mag() / reach)
    return direction.set_mag(magnitude)


def attract_and_repel(
    source: Vector,
    source_mass: float,
    target: Vector,
    target_mass: float,
    rest_distance: float = 60,
    g: float = 1.0,
) -> Vector:
    """Exercise 2.13: attracts from far away, repels up close.

    ``G * m1 * m2 * (d - rest) / d³``: gravity's inverse-square falloff, times a factor that
    changes sign at ``rest_distance``, where the two balance.
    """
    direction = source - target
    distance = max(direction.mag(), 5.0)
    strength = g * source_mass * target_mass * (distance - rest_distance) / distance**3
    return direction.set_mag(strength) if strength >= 0 else -direction.set_mag(-strength)


def attract_all(
    bodies: Sequence[Mover],
    g: float = 1.0,
    distance_range: tuple[float, float] = (5.0, 25.0),
) -> None:
    """Every body attracts every other one (Example 2.9): n * (n - 1) forces per frame."""
    for body in bodies:
        for other in bodies:
            if other is not body:
                force = attraction(
                    other.position, other.mass, body.position, body.mass, g, distance_range
                )
                body.apply_force(force)


# The figure-eight solution of the three-body problem (Chenciner and Montgomery, 2000), with
# G = m = 1: three equal masses chasing each other along one figure eight.
FIGURE_EIGHT_POSITION = Vector(0.97000436, -0.24308753)
FIGURE_EIGHT_VELOCITY = Vector(-0.93240737, -0.86473146)
FIGURE_EIGHT_PERIOD = 6.32591398


def figure_eight(center: Vector, scale: float, period_frames: float) -> tuple[Mover, ...]:
    """Exercise 2.14: three bodies set up for the figure-eight choreography.

    ``scale`` pixels per unit of length. The mass is chosen so one lap takes
    ``period_frames`` frames with G = 1: time scales as ``L^1.5 / sqrt(G * m)``.
    """
    time_scale = period_frames / FIGURE_EIGHT_PERIOD  # frames per unit of time
    mass = scale**3 / time_scale**2
    speed = scale / time_scale
    p, v = FIGURE_EIGHT_POSITION, FIGURE_EIGHT_VELOCITY
    return (
        Mover(center + p * scale, v * (-speed / 2), mass),
        Mover(center - p * scale, v * (-speed / 2), mass),
        Mover(center, v * speed, mass),
    )


@dataclass
class Galaxy:
    """Exercise 2.16's solution: a heavy sun at the origin and stars in near-circular orbits.

    With 100 stars there are ~10,000 attractions per frame, too many for one ``Vector`` at a
    time, so the state lives in numpy arrays: row ``i`` is star ``i``.
    """

    positions: FloatArray
    velocities: FloatArray
    masses: FloatArray
    sun_mass: float = 500.0

    @classmethod
    def spiral(cls, rng: random.Random, count: int = 100) -> Galaxy:
        positions, velocities, masses = [], [], []
        for _ in range(count):
            direction = Vector.random2d(rng)
            positions.append((direction * rng.uniform(100, 150)).xy)
            velocities.append((direction.rotate(math.pi / 2) * rng.uniform(10, 15)).xy)
            masses.append(rng.uniform(10, 15))
        return cls(np.array(positions), np.array(velocities), np.array(masses))

    def step(self) -> None:
        # offsets[i, j] points from star i to star j; the squared distance is clamped to
        # [100, 1000], and the force's magnitude is m_i * m_j / d².
        offsets = self.positions[None, :, :] - self.positions[:, None, :]
        dist_sq = (offsets**2).sum(axis=2)
        lengths = np.sqrt(dist_sq)
        np.fill_diagonal(lengths, 1.0)  # a star doesn't attract itself (offset is 0 anyway)
        strength = self.masses[None, :] / np.clip(dist_sq, 100, 1000)  # force / m_i
        accelerations = (offsets / lengths[:, :, None] * strength[:, :, None]).sum(axis=1)
        # The sun sits at the origin.
        to_sun = -self.positions
        sun_dist_sq = (to_sun**2).sum(axis=1)
        sun_strength = self.sun_mass / np.clip(sun_dist_sq, 100, 1000)
        accelerations += to_sun / np.sqrt(sun_dist_sq)[:, None] * sun_strength[:, None]
        self.velocities += accelerations
        self.positions += self.velocities
