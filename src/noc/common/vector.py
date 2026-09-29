"""A 2D Euclidean vector: p5's ``p5.Vector``, as an immutable value with operators.

``p5.Vector`` methods change the vector they're called on (``position.add(velocity)``), and
the book needs *static* versions (``p5.Vector.sub(a, b)``) whenever the inputs must survive.
Here every operation returns a new vector, so there is only one version of each and no
accidental aliasing:

=================================  ==============================
p5.js                              Python
=================================  ==============================
``position.add(velocity)``         ``position += velocity``
``p5.Vector.sub(mouse, center)``   ``mouse - center``
``v.mult(2)`` / ``v.div(2)``       ``v * 2`` / ``v / 2``
``v.mag()`` / ``v.magSq()``        ``v.mag()`` / ``v.mag_sq()``
``v.normalize()``                  ``v.normalize()``
``v.limit(max)`` / ``v.setMag(m)`` ``v.limit(max)`` / ``v.set_mag(m)``
``v.heading()``                    ``v.heading()``
``p5.Vector.fromAngle(a)``         ``Vector.from_angle(a)``
``p5.Vector.random2D()``           ``Vector.random2d()``
=================================  ==============================
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vector:
    x: float = 0.0
    y: float = 0.0

    # --- construction -------------------------------------------------------------------
    @classmethod
    def from_angle(cls, angle: float, length: float = 1.0) -> Vector:
        """A vector pointing at ``angle`` radians (0 is along +x, clockwise on screen)."""
        return cls(length * math.cos(angle), length * math.sin(angle))

    @classmethod
    def random2d(cls, rng: random.Random | None = None) -> Vector:
        """A unit vector in a random direction."""
        return cls.from_angle((rng or random).uniform(0, 2 * math.pi))

    # --- arithmetic ---------------------------------------------------------------------
    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector:
        return Vector(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vector:
        return Vector(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vector:
        return Vector(-self.x, -self.y)

    def __iter__(self) -> Iterator[float]:
        """Unpacks as ``x, y``."""
        yield self.x
        yield self.y

    @property
    def xy(self) -> tuple[float, float]:
        """The vector as a point tuple, e.g. ``canvas.circle(*position.xy, 48)``."""
        return (self.x, self.y)

    def __bool__(self) -> bool:
        return self.x != 0 or self.y != 0

    # --- length and direction -----------------------------------------------------------
    def mag(self) -> float:
        return math.hypot(self.x, self.y)

    def mag_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalize(self) -> Vector:
        """The unit vector in the same direction; the zero vector stays zero (as in p5)."""
        length = self.mag()
        return self / length if length else self

    def set_mag(self, length: float) -> Vector:
        return self.normalize() * length

    def limit(self, maximum: float) -> Vector:
        """This vector, shortened to ``maximum`` if it's longer."""
        if self.mag_sq() > maximum * maximum:
            return self.set_mag(maximum)
        return self

    def heading(self) -> float:
        """The angle of rotation, in radians, from the +x axis."""
        return math.atan2(self.y, self.x)

    def rotate(self, angle: float) -> Vector:
        cos, sin = math.cos(angle), math.sin(angle)
        return Vector(self.x * cos - self.y * sin, self.x * sin + self.y * cos)

    # --- products and distances ---------------------------------------------------------
    def dot(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y

    def dist(self, other: Vector) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle_between(self, other: Vector) -> float:
        """The unsigned angle between two vectors, from 0 to pi."""
        lengths = self.mag() * other.mag()
        if not lengths:
            return 0.0
        return math.acos(max(-1.0, min(1.0, self.dot(other) / lengths)))

    def lerp(self, other: Vector, amount: float) -> Vector:
        return self + (other - self) * amount
