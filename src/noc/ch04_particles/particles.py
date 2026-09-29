"""Particles, emitters that keep lists of them, and forces acting on whole systems."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from noc.common.mathutils import clamp
from noc.common.physics import Mover
from noc.common.vector import Vector


@dataclass
class Particle(Mover):
    """A mover with a lifespan that counts down every frame (Example 4.1).

    ``lifespan`` doubles as the particle's opacity, fading it out as it dies.
    """

    lifespan: float = 255.0
    decay: float = 2.0

    def update(self) -> None:
        super().update()
        self.lifespan -= self.decay

    @property
    def is_dead(self) -> bool:
        return self.lifespan < 0

    @property
    def alpha(self) -> int:
        return int(clamp(self.lifespan, 0, 255))


@dataclass
class Confetti(Particle):
    """Example 4.5: a particle drawn as a square that spins with its x-position.

    Only the drawing differs from :class:`Particle`, so the subclass adds no behavior; the
    sketches tell them apart with ``match``.
    """


@dataclass
class TexturedParticle(Particle):
    """Exercise 4.12: a particle that remembers which of several textures to draw."""

    texture: int = 0


def falling(position: Vector, rng: random.Random) -> Particle:
    """The book's default particle: a random push upward and sideways (Example 4.1)."""
    return Particle(position, Vector(rng.uniform(-1, 1), rng.uniform(-1, 0)))


def smoke(position: Vector, rng: random.Random) -> Particle:
    """Example 4.8: rises with normally distributed velocities and a short life."""
    velocity = Vector(rng.gauss(0, 0.3), rng.gauss(-1, 0.3))
    return Particle(position, velocity, lifespan=100.0, decay=2.0)


type ParticleFactory = Callable[[Vector, random.Random], Particle]


@dataclass
class Emitter:
    """A particle system: adds particles at its origin and removes the dead ones (Example 4.3).

    ``budget`` (Exercise 4.5) caps how many particles it will ever emit; once they're all dead
    the emitter is :attr:`finished`.
    """

    origin: Vector
    factory: ParticleFactory = falling
    rng: random.Random = field(default_factory=random.Random)
    budget: int | None = None
    particles: list[Particle] = field(default_factory=list)
    emitted: int = 0

    def add_particle(self, count: int = 1) -> None:
        """Exercise 4.13: ``count`` particles at once, one by default."""
        for _ in range(count):
            if self.budget is not None and self.emitted >= self.budget:
                return
            self.particles.append(self.factory(self.origin, self.rng))
            self.emitted += 1

    def apply_force(self, force: Vector) -> None:
        """Example 4.6: the same force on every particle."""
        for particle in self.particles:
            particle.apply_force(force)

    def apply(self, field_force: Callable[[Particle], Vector]) -> None:
        """Example 4.7: a force that depends on each particle (a repeller, an attractor)."""
        for particle in self.particles:
            particle.apply_force(field_force(particle))

    def run(self) -> None:
        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if not p.is_dead]

    @property
    def finished(self) -> bool:
        return self.budget is not None and self.emitted >= self.budget and not self.particles


@dataclass
class PointForce:
    """Example 4.7's repeller, and Exercise 4.9's attractor: an inverse-square push or pull.

    Positive ``power`` repels, negative attracts; the distance is clamped to [5, 50].
    """

    position: Vector
    power: float = 150.0

    def __call__(self, particle: Particle) -> Vector:
        direction = self.position - particle.position
        distance = clamp(direction.mag(), 5, 50)
        return direction.set_mag(-self.power / (distance * distance))


@dataclass
class Shard(Particle):
    """Exercise 4.6: one piece of a shattering block; ``size`` is its side."""

    size: float = 10.0
    drag: float = 0.95

    def update(self) -> None:
        super().update()
        self.velocity *= self.drag


def block(corner: Vector, cols: int, rows: int, size: float) -> list[Shard]:
    """A grid of shards that looks like one solid block until it shatters."""
    return [
        Shard(corner + Vector(i * size, j * size), lifespan=float("inf"), size=size)
        for j in range(rows)
        for i in range(cols)
    ]


def shatter(shards: list[Shard], rng: random.Random, strength: float = 10.0) -> None:
    """Every shard gets a kick in a random direction; they fade from now on."""
    for shard in shards:
        shard.apply_force(Vector.random2d(rng) * strength)
        shard.lifespan = 255.0
