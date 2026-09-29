"""Uniform, normal and custom distributions of random numbers."""

from __future__ import annotations

import colorsys
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Histogram:
    """Counts how often each of ``bins`` equal slices of [0, 1) is picked (Examples 0.2, 0.5)."""

    bins: int
    counts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.counts = [0] * self.bins

    def add(self, value: float) -> None:
        self.counts[min(int(value * self.bins), self.bins - 1)] += 1


def accept_reject(probability: Callable[[float], float], rng: random.Random) -> float:
    """A number in [0, 1) picked with likelihood ``probability(x)`` (Example 0.5).

    Pick ``r1``, then a *qualifying* ``r2``: keep ``r1`` only if ``r2 < probability(r1)``,
    otherwise start over. ``probability`` must return values in [0, 1].
    """
    while True:
        r1 = rng.random()
        if rng.random() < probability(r1):
            return r1


def normal_pdf(x: float, mean: float = 0.0, sd: float = 1.0) -> float:
    """The bell curve (Figure 0.2): the normal distribution's probability density."""
    return math.exp(-((x - mean) ** 2) / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi))


type RGB = tuple[int, int, int]


@dataclass
class Splatter:
    """Exercise 0.4: paint dots, normally distributed in position, size and color.

    Positions are relative to the center of the canvas, in units of half its height.
    """

    spread: float = 0.25
    size: float = 20.0  # mean diameter, in pixels
    size_spread: float = 0.01
    hue: float = 250.0  # degrees
    hue_spread: float = 15.0

    def dot(self, rng: random.Random, canvas_height: float) -> tuple[float, float, float, RGB]:
        """``(x, y, diameter, rgb)`` of a new dot."""
        x = rng.gauss(0, self.spread)
        y = rng.gauss(0, self.spread)
        diameter = max(rng.gauss(self.size / canvas_height, self.size_spread), 0.001)
        hue = rng.gauss(self.hue, self.hue_spread) % 360
        saturation = min(rng.gauss(80, 20), 100)
        brightness = min(rng.gauss(80, 20), 100)
        r, g, b = colorsys.hsv_to_rgb(hue / 360, max(saturation, 0) / 100, max(brightness, 0) / 100)
        return x, y, diameter, (round(r * 255), round(g * 255), round(b * 255))
