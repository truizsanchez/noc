"""Random walkers: an object that takes a random step every frame.

The book writes a new ``Walker`` class for each variation. Here there is one
:class:`Walker`, and what changes between examples is its *step rule*: a function that, given
a random-number generator and the walker, returns the next ``(dx, dy)``.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from noc.ch00_randomness.distributions import accept_reject
from noc.common.mathutils import clamp, remap
from noc.common.noise import Noise, noise

type Point = tuple[float, float]
type StepRule = Callable[[random.Random, Walker], Point]


@dataclass
class Walker:
    x: float
    y: float
    rule: StepRule
    rng: random.Random = field(default_factory=random.Random)
    bounds: Point | None = None  # (width, height) to stay inside, as in Example 0.3

    def step(self) -> None:
        dx, dy = self.rule(self.rng, self)
        self.x += dx
        self.y += dy
        if self.bounds is not None:
            width, height = self.bounds
            self.x = clamp(self.x, 0, width - 1)
            self.y = clamp(self.y, 0, height - 1)


def four_directions(rng: random.Random, walker: Walker) -> Point:
    """Example 0.1: one pixel right, left, down or up, each with the same probability."""
    match rng.randrange(4):
        case 0:
            return (1, 0)
        case 1:
            return (-1, 0)
        case 2:
            return (0, 1)
        case _:
            return (0, -1)


def tends_right(rng: random.Random, walker: Walker) -> Point:
    """Example 0.3: 40% right, 20% each for left, down and up."""
    r = rng.random()
    if r < 0.4:
        return (1, 0)
    if r < 0.6:
        return (-1, 0)
    if r < 0.8:
        return (0, 1)
    return (0, -1)


def skewed(rng: random.Random, walker: Walker) -> Point:
    """Exercise 0.1: any step from -2.75 to 3 on each axis, so down and right win slightly."""
    return (rng.uniform(-2.75, 3), rng.uniform(-2.75, 3))


def toward(target: Callable[[], Point], chance: float = 0.5) -> StepRule:
    """Exercise 0.3: with probability ``chance``, one pixel toward ``target()`` (the mouse)."""

    def rule(rng: random.Random, walker: Walker) -> Point:
        r = rng.random()
        if r >= chance:
            return four_directions(rng, walker)
        tx, ty = target()
        if r < chance / 2:
            return (1 if walker.x < tx else -1, 0)
        return (0, 1 if walker.y < ty else -1)

    return rule


def gaussian(sd: float = 3.0) -> StepRule:
    """Exercise 0.5: a Gaussian random walk, step sizes normally distributed around 0."""

    def rule(rng: random.Random, walker: Walker) -> Point:
        return (rng.gauss(0, sd), rng.gauss(0, sd))

    return rule


def quadratic(size: float = 5.0) -> StepRule:
    """Exercise 0.6: step lengths from accept-reject with probability ``x**2``.

    Long steps are the likeliest, a rough take on a Lévy flight's occasional long jumps.
    """

    def rule(rng: random.Random, walker: Walker) -> Point:
        dx = accept_reject(lambda x: x * x, rng) * size
        dy = accept_reject(lambda x: x * x, rng) * size
        return (dx * rng.choice((-1, 1)), dy * rng.choice((-1, 1)))

    return rule


@dataclass
class NoiseWalker:
    """Example 0.6: position straight from 1D noise, reading two separate parts of it."""

    width: float
    height: float
    noise: Noise = noise
    tx: float = 0.0
    ty: float = 10_000.0  # far from tx, so x and y don't move in lockstep
    x: float = 0.0
    y: float = 0.0

    def step(self) -> None:
        self.x = remap(self.noise(self.tx), 0, 1, 0, self.width)
        self.y = remap(self.noise(self.ty), 0, 1, 0, self.height)
        self.tx += 0.01
        self.ty += 0.01


def noise_steps(noise: Noise = noise) -> StepRule:
    """Exercise 0.7: noise decides the step on each axis (-1 to 1) instead of the position."""
    offsets = [0.0, 10_000.0]

    def rule(rng: random.Random, walker: Walker) -> Point:
        dx = remap(noise(offsets[0]), 0, 1, -1, 1)
        dy = remap(noise(offsets[1]), 0, 1, -1, 1)
        offsets[0] += 0.01
        offsets[1] += 0.01
        return (dx, dy)

    return rule
