"""The ``Mover``: position, velocity and acceleration, and ways of choosing the acceleration.

As with chapter 0's walkers, the book's examples differ only in how the acceleration is
picked each frame, so those rules are functions and :class:`Mover` stays the same.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from noc.common.mathutils import remap
from noc.common.noise import Noise
from noc.common.vector import Vector


@dataclass
class Mover:
    """Motion 101: velocity changes by the acceleration, position by the velocity."""

    position: Vector
    velocity: Vector = Vector()
    acceleration: Vector = Vector()
    top_speed: float = math.inf

    def update(self) -> None:
        self.velocity = (self.velocity + self.acceleration).limit(self.top_speed)
        self.position += self.velocity

    def wrap_edges(self, width: float, height: float) -> None:
        """Leave one side of the canvas, come back on the other (``checkEdges()``)."""
        x, y = self.position
        if x > width:
            x = 0
        elif x < 0:
            x = width
        if y > height:
            y = 0
        elif y < 0:
            y = height
        self.position = Vector(x, y)

    def bounce_edges(self, width: float, height: float) -> None:
        """Reverse the velocity on each axis where the position left the canvas (Example 1.2)."""
        vx, vy = self.velocity
        if not 0 <= self.position.x <= width:
            vx = -vx
        if not 0 <= self.position.y <= height:
            vy = -vy
        self.velocity = Vector(vx, vy)


def random_acceleration(rng: random.Random, maximum: float = 2.0) -> Vector:
    """Example 1.9: a random direction and a random strength up to ``maximum``."""
    return Vector.random2d(rng) * rng.uniform(0, maximum)


def toward(target: Vector, position: Vector, strength: float = 0.2) -> Vector:
    """Example 1.10: a constant-strength acceleration pointing at ``target``."""
    return (target - position).set_mag(strength)


def toward_by_distance(
    target: Vector, position: Vector, max_distance: float, strength: float = 0.2
) -> Vector:
    """Exercise 1.8: like :func:`toward`, but stronger the farther away the target is."""
    direction = target - position
    return direction.set_mag(remap(direction.mag(), 0, max_distance, 0, strength))


@dataclass
class NoiseAcceleration:
    """Exercise 1.6: each axis of the acceleration follows its own stretch of Perlin noise."""

    noise: Noise
    strength: float = 0.1
    offsets: list[float] = field(default_factory=lambda: [0.0, 10_000.0])

    def __call__(self) -> Vector:
        ax = remap(self.noise(self.offsets[0]), 0, 1, -1, 1)
        ay = remap(self.noise(self.offsets[1]), 0, 1, -1, 1)
        self.offsets = [t + 0.01 for t in self.offsets]
        return Vector(ax, ay) * self.strength


@dataclass
class Train:
    """Exercise 1.5: arrow keys change the acceleration; it can brake but not reverse."""

    position: Vector
    velocity: Vector = Vector(1, 0)
    acceleration: float = 0.0
    top_speed: float = 25.0

    def throttle(self, delta: float) -> None:
        self.acceleration += delta

    def update(self, width: float, length: float) -> None:
        self.position += self.velocity
        speed = min(self.velocity.x + self.acceleration, self.top_speed)
        if speed <= 0:
            speed, self.acceleration = 0.0, 0.0
        self.velocity = Vector(speed, 0)
        if self.position.x > width:
            self.position = Vector(-length, self.position.y)
