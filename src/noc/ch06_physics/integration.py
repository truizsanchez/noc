"""A Brief Interlude: Integration Methods, as numbers instead of words.

Three ways to advance a body one step under an acceleration ``a(position)``:

- **Explicit Euler** moves with the *old* velocity, then updates the velocity.
- **Semi-implicit (symplectic) Euler** updates the velocity first and moves with the new one.
  That's the book's ``velocity.add(acceleration); position.add(velocity)``: what the book
  calls Euler is already this better variant, the one Box2D uses.
- **Verlet** stores the previous position instead of a velocity:
  ``next = 2 * current - previous + a``.

On an orbit, explicit Euler gains energy every step and spirals outward; the other two keep
the orbit closed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from noc.common.vector import Vector

type Acceleration = Callable[[Vector], Vector]


@dataclass
class Body:
    position: Vector
    velocity: Vector


def explicit_euler(body: Body, acceleration: Acceleration) -> None:
    a = acceleration(body.position)
    body.position += body.velocity
    body.velocity += a


def semi_implicit_euler(body: Body, acceleration: Acceleration) -> None:
    body.velocity += acceleration(body.position)
    body.position += body.velocity


@dataclass
class VerletBody:
    position: Vector
    previous: Vector

    @classmethod
    def launched(cls, position: Vector, velocity: Vector) -> VerletBody:
        return cls(position, position - velocity)

    @property
    def velocity(self) -> Vector:
        return self.position - self.previous


def verlet(body: VerletBody, acceleration: Acceleration) -> None:
    current = body.position
    body.position = current * 2 - body.previous + acceleration(current)
    body.previous = current


def gravity_toward(center: Vector, gm: float) -> Acceleration:
    """``GM / r²`` toward ``center``."""

    def acceleration(position: Vector) -> Vector:
        offset = center - position
        return offset.set_mag(gm / offset.mag_sq())

    return acceleration


def orbital_energy(position: Vector, velocity: Vector, center: Vector, gm: float) -> float:
    """Kinetic plus potential energy per unit mass: constant for a true orbit."""
    return velocity.mag_sq() / 2 - gm / position.dist(center)
