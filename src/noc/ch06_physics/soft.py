"""Soft bodies built twice: with our Verlet engine and with pymunk's damped springs.

A :class:`Topology` describes a soft body (particles, springs, which particles are pinned);
:class:`VerletModel` and :class:`PymunkModel` simulate it. The two differ in kind:

- Verlet springs are position *constraints* relaxed many times per frame (50 in Toxiclibs):
  unconditionally stable, and as stiff as the strength and the iteration count allow. A
  hanging cloth still stretches near its pins when gravity outpulls the relaxation.
- pymunk's ``DampedSpring`` is a *force* (Hooke's law plus damping) on point masses. It
  needs its stiffness tuned per model and several substeps per frame to stay stable, since
  a stiff force overshoots with large steps; with those, it holds its shape well, and being
  compiled C it's also faster than the numpy engine.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import pymunk

from noc.ch06_physics.verlet import VerletPhysics

type Point = tuple[float, float]


@dataclass(frozen=True)
class SpringSpec:
    a: int
    b: int
    rest_length: float
    strength: float  # Verlet relaxation fraction


@dataclass
class Topology:
    points: list[Point]
    springs: list[SpringSpec]
    pinned: set[int] = field(default_factory=set)
    gravity: float = 0.5
    bounds: tuple[float, float, float, float] | None = None
    pymunk_stiffness: float = 5.0  # per unit of mass
    pymunk_damping: float = 0.2


class SoftModel(Protocol):
    name: str
    topology: Topology
    step_ms: float

    def step(self) -> None: ...

    def points(self) -> list[Point]: ...

    def drag(self, index: int, x: float, y: float) -> None: ...


def _timed(step: Callable[[], None], model: SoftModel) -> None:
    start = time.perf_counter()
    step()
    elapsed = (time.perf_counter() - start) * 1000
    model.step_ms = elapsed if not model.step_ms else 0.9 * model.step_ms + 0.1 * elapsed


class VerletModel:
    name = "Verlet (our engine)"

    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.step_ms = 0.0
        self.physics = VerletPhysics(gravity=(0, topology.gravity), bounds=topology.bounds)
        for x, y in topology.points:
            self.physics.add_particle(x, y)
        for s in topology.springs:
            self.physics.add_spring(s.a, s.b, s.rest_length, s.strength)
        for i in topology.pinned:
            self.physics.lock(i)

    def step(self) -> None:
        _timed(self.physics.update, self)

    def points(self) -> list[Point]:
        return [(float(x), float(y)) for x, y in self.physics.positions]

    def drag(self, index: int, x: float, y: float) -> None:
        self.physics.move_to(index, x, y)


class PymunkModel:
    name = "pymunk DampedSpring"

    def __init__(self, topology: Topology, substeps: int = 10) -> None:
        self.topology = topology
        self.substeps = substeps
        self.step_ms = 0.0
        self.space = pymunk.Space()
        self.space.gravity = (0, topology.gravity)
        self.space.damping = 0.99
        self.bodies: list[pymunk.Body] = []
        no_self_collisions = pymunk.ShapeFilter(group=1)
        for i, (x, y) in enumerate(topology.points):
            if i in topology.pinned:
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
                body.position = (x, y)
                self.space.add(body)
            else:
                body = pymunk.Body(1, float("inf"))  # a point mass: it never rotates
                body.position = (x, y)
                shape = pymunk.Circle(body, 2)
                shape.filter = no_self_collisions
                self.space.add(body, shape)
            self.bodies.append(body)
        for s in topology.springs:
            spring = pymunk.DampedSpring(
                self.bodies[s.a],
                self.bodies[s.b],
                (0, 0),
                (0, 0),
                s.rest_length,
                topology.pymunk_stiffness,
                topology.pymunk_damping,
            )
            self.space.add(spring)
        if topology.bounds is not None:
            left, top, right, bottom = topology.bounds
            corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
            for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
                self.space.add(pymunk.Segment(self.space.static_body, a, b, 1))

    def step(self) -> None:
        def advance() -> None:
            for _ in range(self.substeps):
                self.space.step(1 / self.substeps)

        _timed(advance, self)

    def points(self) -> list[Point]:
        return [(float(b.position.x), float(b.position.y)) for b in self.bodies]

    def drag(self, index: int, x: float, y: float) -> None:
        body = self.bodies[index]
        body.position = (x, y)
        body.velocity = (0, 0)
        self.space.reindex_shapes_for_body(body)


# --- the book's soft bodies ----------------------------------------------------------------
def string(width: float, height: float, total: int = 20, spacing: float = 10) -> Topology:
    """Example 6.12: a chain of particles, the first one pinned."""
    points = [(width / 2 + i * spacing, 0.0) for i in range(total)]
    springs = [SpringSpec(i, i + 1, spacing, 0.2) for i in range(total - 1)]
    return Topology(
        points, springs, {0}, 0.5, (0, 0, width, height), pymunk_stiffness=40, pymunk_damping=1
    )


def soft_body(width: float, height: float) -> Topology:
    """Example 6.13: a hexagon-ish character with cross braces."""
    points = [(200.0, 25.0), (400, 25), (350, 125), (400, 225), (200, 225), (250, 125)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (5, 2), (0, 3), (1, 4)]

    def length(a: int, b: int) -> float:
        (ax, ay), (bx, by) = points[a], points[b]
        return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    springs = [SpringSpec(a, b, length(a, b), 0.01) for a, b in pairs]
    return Topology(
        points, springs, set(), 0.5, (0, 0, width, height), pymunk_stiffness=2, pymunk_damping=0.5
    )


def cloth(cols: int = 65, rows: int = 20, spacing: float = 10) -> Topology:
    """Exercise 6.10: a grid with horizontal and vertical springs, pinned every fourth column.

    The solution starts every row at y = 0 (its ``y = y`` never advances); here the rows start
    ``spacing`` apart, so the springs begin at rest.
    """
    points = [(i * spacing, j * spacing) for i in range(cols) for j in range(rows)]

    def index(i: int, j: int) -> int:
        return i * rows + j

    springs = [
        SpringSpec(index(i, j), index(i + di, j + dj), spacing, 0.25)
        for i in range(cols)
        for j in range(rows)
        for di, dj in ((1, 0), (0, 1))
        if i + di < cols and j + dj < rows
    ]
    pinned = {index(i, 0) for i in range(0, cols, 4)}
    return Topology(points, springs, pinned, 1.0, None, pymunk_stiffness=60, pymunk_damping=2)
