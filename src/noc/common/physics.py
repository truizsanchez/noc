"""Newton's second law for the book's movers, and the forces of chapter 2.

A :class:`Mover` accumulates forces during a frame (``apply_force``), turns them into an
acceleration (``F = m * a``, so ``a = F / m``), and in :meth:`Mover.update` adds the
acceleration to the velocity and the velocity to the position (Euler integration, one step per
frame). The forces are plain functions that return a :class:`~noc.common.vector.Vector`, so
the same mover can feel any combination of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from noc.common.mathutils import clamp
from noc.common.vector import Vector


@dataclass
class Mover:
    position: Vector
    velocity: Vector = Vector()
    mass: float = 1.0
    acceleration: Vector = Vector()
    top_speed: float = math.inf

    def apply_force(self, force: Vector) -> None:
        """Newton's second law with force accumulation: ``a += F / m``."""
        self.acceleration += force / self.mass

    def update(self) -> None:
        self.velocity = (self.velocity + self.acceleration).limit(self.top_speed)
        self.position += self.velocity
        self.acceleration = Vector()  # forces are applied anew every frame

    def bounce(
        self,
        width: float,
        height: float,
        radius: float = 0.0,
        *,
        restitution: float = 1.0,
        top: bool = False,
    ) -> None:
        """Keep the circle inside the walls and floor (and ceiling, with ``top``).

        On contact, the position is put back at the edge and the velocity along that axis is
        reversed and scaled by ``restitution`` (0.9 loses 10% of the speed per bounce).
        """
        (x, y), (vx, vy) = self.position, self.velocity
        if x > width - radius:
            x, vx = width - radius, -vx * restitution
        elif x < radius:
            x, vx = radius, -vx * restitution
        if y > height - radius:
            y, vy = height - radius, -vy * restitution
        elif top and y < radius:
            y, vy = radius, -vy * restitution
        self.position, self.velocity = Vector(x, y), Vector(vx, vy)


# --- forces --------------------------------------------------------------------------------
def weight(mass: float, gravity: Vector = Vector(0, 0.1)) -> Vector:
    """Gravity scaled by mass, so every object falls with the same acceleration (Example 2.3)."""
    return gravity * mass


def friction(velocity: Vector, coefficient: float, normal: float = 1.0) -> Vector:
    """Friction: ``-μN`` times the unit velocity, opposing the motion (Example 2.4)."""
    return velocity.normalize() * (-coefficient * normal)


def drag(velocity: Vector, coefficient: float, *, mass: float | None = None) -> Vector:
    """Fluid drag: ``-c * v²`` against the velocity (Example 2.5).

    With a large coefficient, one frame's worth of drag can be bigger than the momentum it
    opposes and push the object backward. Passing the ``mass`` limits the force so that it
    can at most stop the object, never reverse it (Exercise 2.8).
    """
    speed = velocity.mag()
    force = velocity.normalize() * (-coefficient * speed * speed)
    if mass is not None:
        force = force.limit(speed * mass)
    return force


def attraction(
    source: Vector,
    source_mass: float,
    target: Vector,
    target_mass: float,
    g: float = 1.0,
    distance_range: tuple[float, float] = (5.0, 25.0),
) -> Vector:
    """Gravitational attraction of ``target`` toward ``source``: ``G * m1 * m2 / d²``.

    The distance is clamped to ``distance_range``, as in the book: the lower bound avoids huge
    forces when the objects overlap, the upper one keeps faraway objects from barely moving.
    """
    direction = source - target
    distance = clamp(direction.mag(), *distance_range)
    return direction.set_mag(g * source_mass * target_mass / (distance * distance))


@dataclass
class Liquid:
    """A rectangle of fluid that drags whatever is inside it (Example 2.5)."""

    x: float
    y: float
    width: float
    height: float
    coefficient: float

    def contains(self, position: Vector) -> bool:
        return (
            self.x < position.x < self.x + self.width and self.y < position.y < self.y + self.height
        )
