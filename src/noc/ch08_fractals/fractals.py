"""Fractals as geometry: recursive generators that yield shapes instead of drawing them.

The book's recursive functions draw as they go (``circle()``, ``line()``). Here each
recursion *yields* what it would draw, so the same fractal can be drawn, counted, animated
(drawn a piece at a time) or tested without a window.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from noc.common.mathutils import remap
from noc.common.noise import Noise
from noc.common.vector import Vector

type Segment = tuple[Vector, Vector]


# --- recursion -----------------------------------------------------------------------------
def nested_circles(x: float, y: float, r: float) -> Iterator[tuple[float, float, float]]:
    """Example 8.1: a circle, then the same circle 75% the size, down to a radius of 4."""
    yield x, y, r
    if r > 4:
        yield from nested_circles(x, y, r * 0.75)


def circles_twice(x: float, y: float, r: float) -> Iterator[tuple[float, float, float]]:
    """Example 8.2: each circle spawns two half-size circles on its left and right."""
    yield x, y, r
    if r > 4:
        yield from circles_twice(x + r / 2, y, r / 2)
        yield from circles_twice(x - r / 2, y, r / 2)


def circles_four_times(x: float, y: float, r: float) -> Iterator[tuple[float, float, float]]:
    """Example 8.3: four half-size circles, left, right, above and below."""
    yield x, y, r
    if r > 16:
        for dx, dy in ((r / 2, 0), (-r / 2, 0), (0, r / 2), (0, -r / 2)):
            yield from circles_four_times(x + dx, y + dy, r / 2)


def cross_lines(a: Vector, b: Vector) -> Iterator[Segment]:
    """Exercise 8.1's solution: every line grows two perpendicular lines at its ends, each
    two-thirds of its length, alternating horizontal and vertical."""
    yield a, b
    d = b - a
    if d.x == 0 and d.y > 4:
        for end in (a, b):
            yield from cross_lines(end - Vector(d.y / 3, 0), end + Vector(d.y / 3, 0))
    elif d.y == 0 and d.x > 4:
        for end in (a, b):
            yield from cross_lines(end - Vector(0, d.x / 3), end + Vector(0, d.x / 3))


def cantor(x: float, y: float, length: float) -> Iterator[tuple[float, float, float]]:
    """Example 8.4: a line, then the Cantor set of its first and last thirds 20 px below."""
    if length > 1:
        yield x, y, length
        yield from cantor(x, y + 20, length / 3)
        yield from cantor(x + 2 * length / 3, y + 20, length / 3)


@dataclass(frozen=True)
class CantorLine:
    """Exercise 8.4: the Cantor set with objects, one generation at a time."""

    x: float
    y: float
    length: float

    def children(self) -> tuple[CantorLine, CantorLine]:
        third = self.length / 3
        return (
            CantorLine(self.x, self.y + 20, third),
            CantorLine(self.x + 2 * third, self.y + 20, third),
        )


def cantor_generations(first: CantorLine, count: int) -> list[list[CantorLine]]:
    generations = [[first]]
    for _ in range(count - 1):
        generations.append([child for line in generations[-1] for child in line.children()])
    return generations


# --- the Koch curve ------------------------------------------------------------------------
@dataclass(frozen=True)
class KochLine:
    """Example 8.5: a segment that knows the five points of its Koch replacement."""

    start: Vector
    end: Vector

    def koch_points(self) -> tuple[Vector, Vector, Vector, Vector, Vector]:
        third = (self.end - self.start) / 3
        a, e = self.start, self.end
        b = a + third
        d = b + third
        c = b + third.rotate(-math.pi / 3)  # the bump, up (y points down)
        return a, b, c, d, e

    def replace(self) -> list[KochLine]:
        a, b, c, d, e = self.koch_points()
        return [KochLine(a, b), KochLine(b, c), KochLine(c, d), KochLine(d, e)]


def koch(lines: list[KochLine], generations: int) -> list[KochLine]:
    for _ in range(generations):
        lines = [piece for line in lines for piece in line.replace()]
    return lines


def snowflake(center: Vector, radius: float, generations: int) -> list[KochLine]:
    """Exercise 8.2: three Koch curves on the sides of a triangle, bumps outward."""
    corners = [center + Vector.from_angle(-math.pi / 2 + k * 2 * math.pi / 3, radius)
               for k in range(3)]  # fmt: skip
    sides = [KochLine(corners[k], corners[(k + 1) % 3]) for k in range(3)]
    return koch(sides, generations)


def sierpinski(a: Vector, b: Vector, c: Vector, depth: int) -> Iterator[tuple[Vector, ...]]:
    """Exercise 8.5: the Sierpiński triangle: three half-size triangles at the corners."""
    if depth == 0:
        yield a, b, c
        return
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    yield from sierpinski(a, ab, ca, depth - 1)
    yield from sierpinski(ab, b, bc, depth - 1)
    yield from sierpinski(ca, bc, c, depth - 1)


# --- trees ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Branch:
    start: Vector
    end: Vector
    length: float


def tree(root: Vector, length: float, angle: float, shrink: float = 0.67) -> Iterator[Branch]:
    """Example 8.6 with vectors instead of ``translate``/``rotate`` (Exercise 8.8's hint):
    every branch splits into two, turned ``angle`` either way, 67% as long."""

    def grow(start: Vector, heading: float, length: float) -> Iterator[Branch]:
        end = start + Vector.from_angle(heading, length)
        yield Branch(start, end, length)
        if length * shrink > 2:
            yield from grow(end, heading + angle, length * shrink)
            yield from grow(end, heading - angle, length * shrink)

    yield from grow(root, -math.pi / 2, length)


def stochastic_tree(root: Vector, length: float, rng: random.Random) -> Iterator[Branch]:
    """Example 8.7: one to three branches at random angles up to 90 degrees each way."""

    def grow(start: Vector, heading: float, length: float) -> Iterator[Branch]:
        end = start + Vector.from_angle(heading, length)
        yield Branch(start, end, length)
        if length * 0.67 > 2:
            for _ in range(rng.randrange(1, 4)):
                yield from grow(
                    end, heading + rng.uniform(-math.pi / 2, math.pi / 2), length * 0.67
                )

    yield from grow(root, -math.pi / 2, length)


def windy_tree(
    root: Vector, length: float, noise: Noise, time: float, rng: random.Random
) -> Iterator[Branch]:
    """Exercise 8.9: branch angles from Perlin noise, moving through time like wind.

    ``rng`` must be reseeded identically every frame, so only the angles change.
    """

    def grow(start: Vector, heading: float, length: float, xoff: float) -> Iterator[Branch]:
        end = start + Vector.from_angle(heading, length)
        yield Branch(start, end, length)
        length *= 0.7
        xoff += 0.1
        if length > 4:
            n = rng.randrange(1, 5)
            for i in range(n):
                theta = remap(noise(xoff + i, time), 0, 1, -math.pi / 2, math.pi / 2)
                if n % 2 == 0:
                    theta = -theta
                yield from grow(end, heading + theta, length, xoff)

    yield from grow(root, -math.pi / 2, length, 0.0)


def branch_weight(length: float) -> float:
    """Exercise 8.7: thick trunk, thin twigs (``map(length, 2, 80, 1, 8)``)."""
    return remap(length, 2, 80, 1, 8)


@dataclass
class GrowingBranch:
    """Exercise 8.8's solution: a branch that grows for a while, then splits in two."""

    start: Vector
    end: Vector
    velocity: Vector
    timer: float
    initial_timer: float = field(init=False)
    growing: bool = True

    def __post_init__(self) -> None:
        self.initial_timer = self.timer

    def update(self) -> None:
        if self.growing:
            self.end += self.velocity

    def time_to_branch(self) -> bool:
        self.timer -= 1
        if self.timer < 0 and self.growing:
            self.growing = False
            return True
        return False

    def split(self, degrees: float) -> GrowingBranch:
        velocity = self.velocity.rotate(math.radians(degrees))
        return GrowingBranch(self.end, self.end, velocity, self.initial_timer * 0.66)


@dataclass
class GrowingTree:
    """Branches that sprout two children when their timer runs out; leaves once full."""

    branches: list[GrowingBranch]
    max_branches: int = 1024
    leaves: list[Vector] = field(default_factory=list)

    @classmethod
    def seed(cls, root: Vector) -> GrowingTree:
        return cls([GrowingBranch(root, root, Vector(0, -1), 80)])

    def update(self) -> None:
        for branch in list(self.branches):
            branch.update()
            if branch.time_to_branch():
                if len(self.branches) < self.max_branches:
                    self.branches += [branch.split(30), branch.split(-25)]
                else:
                    self.leaves.append(branch.end)
