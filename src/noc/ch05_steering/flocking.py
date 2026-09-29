"""Flocking at scale: a numpy flock, and the spatial subdivisions of "Algorithmic Efficiency".

Every boid looks at every other boid: ``n²`` distance checks per behavior per frame. The book
answers with a bin lattice and a quadtree (Examples 5.12 and 5.13), which cut the number of
checks; both are here, working on :class:`~noc.ch05_steering.vehicle.Vehicle` objects. For
the plain flocking example, :class:`Flock` instead does all ``n²`` checks at once with numpy
arrays, which in Python is faster than any per-object loop.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from noc.ch05_steering.vehicle import FlockWeights, Vehicle

type FloatArray = npt.NDArray[np.float64]


def _set_mag(vectors: FloatArray, length: float | FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0) * length


def _limit(vectors: FloatArray, maximum: float) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1.0, np.divide(maximum, norms, out=np.ones_like(norms), where=norms > 0))
    return vectors * scale


@dataclass
class Flock:
    """Example 5.11's rules for a whole flock at once: row ``i`` is boid ``i``."""

    positions: FloatArray
    velocities: FloatArray
    max_speed: float = 3.0
    max_force: float = 0.05
    weights: FlockWeights = field(default_factory=FlockWeights)

    @classmethod
    def at(cls, x: float, y: float, count: int, rng: random.Random) -> Flock:
        velocities = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(count)]
        return cls(np.tile([x, y], (count, 1)).astype(np.float64), np.array(velocities))

    def add(self, x: float, y: float, rng: random.Random) -> None:
        self.positions = np.vstack([self.positions, [x, y]])
        velocity = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
        self.velocities = np.vstack([self.velocities, velocity])

    def steer(self, desired: FloatArray, active: npt.NDArray[np.bool_]) -> FloatArray:
        force = _limit(_set_mag(desired, self.max_speed) - self.velocities, self.max_force)
        return np.where(active[:, None], force, 0.0)

    def forces(self, extra: FloatArray | None = None) -> FloatArray:
        """Separation, alignment and cohesion for every boid (plus ``extra`` steering)."""
        w = self.weights
        offsets = self.positions[:, None, :] - self.positions[None, :, :]  # i minus j
        dist = np.linalg.norm(offsets, axis=2)
        near = (dist > 0) & (dist < w.separation_distance)
        neighbors = (dist > 0) & (dist < w.neighbor_distance)
        count = neighbors.sum(axis=1)

        # Away from each close neighbor, with magnitude 1 / d.
        safe = np.where(dist > 0, dist, 1.0)
        away = (offsets / safe[:, :, None] ** 2 * near[:, :, None]).sum(axis=1)
        separation = self.steer(away, near.any(axis=1))
        alignment = self.steer(neighbors.astype(float) @ self.velocities, count > 0)
        center = (neighbors.astype(float) @ self.positions) / np.maximum(count, 1)[:, None]
        cohesion = self.steer(center - self.positions, count > 0)
        total = w.separation * separation + w.alignment * alignment + w.cohesion * cohesion
        return total if extra is None else total + extra

    def seek(self, target: tuple[float, float]) -> FloatArray:
        """Exercise 5.16: steering toward one target for every boid."""
        return self.steer(np.asarray(target) - self.positions, np.ones(len(self), bool))

    def update(self, forces: FloatArray, width: float, height: float, r: float = 3.0) -> None:
        self.velocities = _limit(self.velocities + forces, self.max_speed)
        self.positions += self.velocities
        # Wrap around the edges, r beyond the canvas.
        for axis, size in enumerate((width, height)):
            column = self.positions[:, axis]
            column[column < -r] = size + r
            column[column > size + r] = -r

    def __len__(self) -> int:
        return len(self.positions)


# --- spatial subdivisions ------------------------------------------------------------------
@dataclass
class BinLattice:
    """Example 5.12: vehicles sorted into square bins; neighbors come from the 3x3 around."""

    resolution: float
    bins: defaultdict[tuple[int, int], list[Vehicle]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def cell(self, vehicle: Vehicle) -> tuple[int, int]:
        return (
            math.floor(vehicle.position.x / self.resolution),
            math.floor(vehicle.position.y / self.resolution),
        )

    def rebuild(self, vehicles: Sequence[Vehicle]) -> None:
        self.bins.clear()
        for vehicle in vehicles:
            self.bins[self.cell(vehicle)].append(vehicle)

    def neighbors(self, vehicle: Vehicle) -> list[Vehicle]:
        col, row = self.cell(vehicle)
        return [
            other
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for other in self.bins.get((col + i, row + j), ())
        ]


@dataclass(frozen=True)
class Rect:
    """An axis-aligned box given by its center and half-sizes, as in the book's quadtree."""

    x: float
    y: float
    w: float
    h: float

    def contains(self, x: float, y: float) -> bool:
        return self.x - self.w <= x < self.x + self.w and self.y - self.h <= y < self.y + self.h

    def intersects(self, other: Rect) -> bool:
        return not (
            other.x - other.w > self.x + self.w
            or other.x + other.w < self.x - self.w
            or other.y - other.h > self.y + self.h
            or other.y + other.h < self.y - self.h
        )


@dataclass
class QuadTree:
    """Example 5.13: each node holds up to ``capacity`` vehicles, then splits into four."""

    boundary: Rect
    capacity: int = 4
    items: list[Vehicle] = field(default_factory=list)
    children: list[QuadTree] = field(default_factory=list)

    def insert(self, vehicle: Vehicle) -> bool:
        if not self.boundary.contains(*vehicle.position.xy):
            return False
        if len(self.items) < self.capacity:
            self.items.append(vehicle)
            return True
        if not self.children:
            self.subdivide()
        return any(child.insert(vehicle) for child in self.children)

    def subdivide(self) -> None:
        b = self.boundary
        w, h = b.w / 2, b.h / 2
        self.children = [
            QuadTree(Rect(b.x + dx * w, b.y + dy * h, w, h), self.capacity)
            for dx, dy in ((1, -1), (-1, -1), (1, 1), (-1, 1))
        ]

    def query(self, area: Rect) -> Iterator[Vehicle]:
        if not self.boundary.intersects(area):
            return
        yield from (v for v in self.items if area.contains(*v.position.xy))
        for child in self.children:
            yield from child.query(area)

    def boxes(self) -> Iterator[Rect]:
        """Every node's boundary, for drawing the subdivision."""
        yield self.boundary
        for child in self.children:
            yield from child.boxes()


# --- lookup tables -------------------------------------------------------------------------
@dataclass
class SinCosTable:
    """Example 5.14: sine and cosine precomputed every ``precision`` degrees."""

    precision: float = 0.5
    sin: list[float] = field(init=False)
    cos: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.period = math.floor(360 / self.precision)
        step = math.radians(self.precision)
        self.sin = [math.sin(i * step) for i in range(self.period)]
        self.cos = [math.cos(i * step) for i in range(self.period)]

    def index(self, degrees: float) -> int:
        return int(degrees / self.precision) % self.period
