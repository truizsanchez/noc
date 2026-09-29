"""L-systems: rewriting a string by rules, then drawing it with a turtle."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from noc.common.vector import Vector

type Rule = str | Sequence[tuple[float, str]]
"""A replacement, or (Exercise 8.12) weighted alternatives picked at random each time."""


@dataclass
class LSystem:
    sentence: str
    rules: Mapping[str, Rule]
    rng: random.Random | None = None

    def replacement(self, char: str) -> str:
        rule = self.rules.get(char)
        if rule is None:
            return char  # symbols without a rule are copied unchanged
        if isinstance(rule, str):
            return rule
        weights, options = zip(*rule, strict=True)
        choice: str = (self.rng or random).choices(options, weights)[0]
        return choice

    def generate(self) -> None:
        self.sentence = "".join(self.replacement(c) for c in self.sentence)


def turtle(
    sentence: str, length: float, angle: float, start: Vector, heading: float = -math.pi / 2
) -> Iterator[tuple[Vector, Vector, int]]:
    """Example 8.9's turtle, with vectors instead of transformations (Exercise 8.11).

    ``F`` draws forward, ``G`` moves forward without drawing, ``+``/``-`` turn, ``[``/``]``
    save and restore the turtle. Yields ``(start, end, depth)`` for every ``F``, where depth
    counts the open brackets (Exercise 8.12's use of it: thinner, lighter twigs).
    """
    position, direction = start, heading
    stack: list[tuple[Vector, float]] = []
    for char in sentence:
        match char:
            case "F" | "G":
                end = position + Vector.from_angle(direction, length)
                if char == "F":
                    yield position, end, len(stack)
                position = end
            case "+":
                direction += angle
            case "-":
                direction -= angle
            case "[":
                stack.append((position, direction))
            case "]":
                position, direction = stack.pop()


@dataclass(frozen=True)
class Preset:
    name: str
    axiom: str
    rules: Mapping[str, Rule]
    angle: float  # degrees
    generations: int
    length: float
    also_forward: str = ""  # letters that also draw a line, like F


PRESETS = [
    Preset("Example 8.9's plant", "F", {"F": "FF+[+F-F-F]-[-F+F+F]"}, 25, 4, 4),
    Preset("Koch curve", "F", {"F": "F+F-F-F+F"}, 90, 4, 2.5),
    Preset("Sierpinski arrowhead", "A", {"A": "B-A-B", "B": "A+B+A"}, 60, 6, 3, "AB"),
    Preset("dragon curve", "FX", {"X": "X+YF+", "Y": "-FX-Y"}, 90, 10, 4),
    Preset(
        "stochastic plant (Exercise 8.12)",
        "F",
        {"F": [(1, "F[+F]F[-F]F"), (1, "F[+F]F"), (1, "F[-F]F")]},
        25.7,
        5,
        2.2,
    ),
]


def expand(preset: Preset, rng: random.Random) -> str:
    system = LSystem(preset.axiom, preset.rules, rng)
    for _ in range(preset.generations):
        system.generate()
    sentence = system.sentence
    for symbol in preset.also_forward:
        sentence = sentence.replace(symbol, "F")
    return sentence
