"""The p5.js transformation stack (``translate``, ``rotate``, ``scale``) as an immutable value.

p5 keeps one mutable matrix and ``push``/``pop`` to save it. Here a :class:`Transform` is a
similarity (translation, rotation, uniform scale), the only kind the book's sketches use, and
:class:`~noc.common.view.Canvas` restores the previous one at the end of a ``with`` block.
Angles are in radians and, as in p5 with y pointing down, positive angles turn clockwise on
screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Transform:
    """Maps local points to canvas points: ``p -> offset + scale * rotate(angle, p)``."""

    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    scale: float = 1.0

    def apply(self, px: float, py: float) -> tuple[float, float]:
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        return (
            self.x + self.scale * (px * cos - py * sin),
            self.y + self.scale * (px * sin + py * cos),
        )

    def translated(self, dx: float, dy: float) -> Transform:
        x, y = self.apply(dx, dy)
        return Transform(x, y, self.angle, self.scale)

    def rotated(self, angle: float) -> Transform:
        return Transform(self.x, self.y, self.angle + angle, self.scale)

    def scaled(self, factor: float) -> Transform:
        return Transform(self.x, self.y, self.angle, self.scale * factor)

    @property
    def is_identity(self) -> bool:
        return self == IDENTITY


IDENTITY = Transform()
