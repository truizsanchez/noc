"""A small Verlet physics engine in the style of Toxiclibs.js's ``VerletPhysics2D``.

Verlet integration keeps each particle's current and previous position instead of a
velocity: ``next = current + (current - previous) + force``. Springs are not forces but
*constraints*: every update, each spring moves its two ends toward (or away from) each other
by a fraction (``strength``) of the error, and this relaxation is repeated ``iterations``
times (Toxiclibs uses 50). That is what makes Verlet springs stable even when stiff: they
correct positions directly, so they can't overshoot the way a stiff force does.

Everything is stored in numpy arrays, one row per particle or spring, so a 1,300-particle
cloth relaxes 2,500 springs 50 times per frame without a Python loop over springs. Toxiclibs
relaxes the springs one at a time, each seeing the corrections of the ones before
(Gauss-Seidel). Relaxing all springs at once from the same positions (Jacobi) would be
simpler but propagates corrections much more slowly, so the springs are split into groups
that share no particle (a greedy edge coloring): each group is relaxed in one numpy step,
and the groups one after another, which is exactly Gauss-Seidel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

EPSILON = 1e-6


@dataclass(frozen=True)
class Attraction:
    """Toxiclibs's ``AttractionBehavior``: pulls (positive) or pushes (negative) particles
    within ``radius`` of particle ``center``, with force ``strength * (1 - d² / radius²)``."""

    center: int
    radius: float
    strength: float


@dataclass
class VerletPhysics:
    gravity: tuple[float, float] = (0.0, 0.0)
    drag: float = 0.0  # fraction of the velocity lost per update
    bounds: tuple[float, float, float, float] | None = None  # left, top, right, bottom
    iterations: int = 50
    positions: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    previous: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    locked: BoolArray = field(default_factory=lambda: np.zeros(0, bool))
    spring_ends: IntArray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    rest_lengths: FloatArray = field(default_factory=lambda: np.zeros(0))
    strengths: FloatArray = field(default_factory=lambda: np.zeros(0))
    min_distance: BoolArray = field(default_factory=lambda: np.zeros(0, bool))
    attractions: list[Attraction] = field(default_factory=list)
    _groups: list[IntArray] | None = None

    # --- building ---------------------------------------------------------------------------
    def add_particle(self, x: float, y: float) -> int:
        """Add a particle at rest; returns its index."""
        self.positions = np.vstack([self.positions, [x, y]])
        self.previous = np.vstack([self.previous, [x, y]])
        self.locked = np.append(self.locked, False)
        return len(self.positions) - 1

    def add_spring(
        self,
        a: int,
        b: int,
        rest_length: float | None = None,
        strength: float = 0.01,
        *,
        min_distance: bool = False,
    ) -> None:
        """A spring between particles ``a`` and ``b`` (``VerletSpring2D``).

        Without ``rest_length``, the current distance is used. ``min_distance`` makes it a
        ``VerletMinDistanceSpring2D``: it only pushes apart, never pulls together.
        """
        if rest_length is None:
            rest_length = float(np.linalg.norm(self.positions[b] - self.positions[a]))
        self.spring_ends = np.vstack([self.spring_ends, [a, b]])
        self.rest_lengths = np.append(self.rest_lengths, rest_length)
        self.strengths = np.append(self.strengths, strength)
        self.min_distance = np.append(self.min_distance, min_distance)
        self._groups = None

    def lock(self, index: int, locked: bool = True) -> None:
        self.locked[index] = locked

    def move_to(self, index: int, x: float, y: float) -> None:
        """Put a particle somewhere and stop it (the book's lock/set/unlock idiom)."""
        self.positions[index] = self.previous[index] = (x, y)

    def clear(self) -> None:
        fresh = VerletPhysics(self.gravity, self.drag, self.bounds, self.iterations)
        self.__dict__.update(fresh.__dict__)

    def __len__(self) -> int:
        return len(self.positions)

    # --- simulation -------------------------------------------------------------------------
    def forces(self) -> FloatArray:
        forces = np.tile(np.asarray(self.gravity, np.float64), (len(self), 1))
        for attraction in self.attractions:
            delta = self.positions[attraction.center] - self.positions
            dist_sq = (delta**2).sum(axis=1)
            inside = (dist_sq < attraction.radius**2) & (dist_sq > 0)
            falloff = 1 - dist_sq / attraction.radius**2
            unit = delta / np.sqrt(np.where(dist_sq > 0, dist_sq, 1))[:, None]
            forces += np.where(inside[:, None], unit * (falloff * attraction.strength)[:, None], 0)
        return forces

    def update(self) -> None:
        free = ~self.locked
        velocity = (self.positions - self.previous + self.forces()) * (1 - self.drag)
        self.previous = np.where(free[:, None], self.positions, self.previous)
        self.positions = np.where(free[:, None], self.positions + velocity, self.positions)
        for _ in range(self.iterations):
            self.relax_springs()
        if self.bounds is not None:
            left, top, right, bottom = self.bounds
            np.clip(self.positions[:, 0], left, right, out=self.positions[:, 0])
            np.clip(self.positions[:, 1], top, bottom, out=self.positions[:, 1])

    def spring_groups(self) -> list[IntArray]:
        """Spring indices split so that no two springs in a group share a particle."""
        if self._groups is None:
            colors: list[set[int]] = []  # particles used by each group
            members: list[list[int]] = []
            for k, (a, b) in enumerate(self.spring_ends.tolist()):
                for used, group in zip(colors, members, strict=True):
                    if a not in used and b not in used:
                        used.update((a, b))
                        group.append(k)
                        break
                else:
                    colors.append({a, b})
                    members.append([k])
            self._groups = [np.array(group, np.int64) for group in members]
        return self._groups

    def relax_springs(self) -> None:
        """One relaxation pass: each spring corrects ``strength`` of its length error."""
        for group in self.spring_groups():
            a, b = self.spring_ends[group, 0], self.spring_ends[group, 1]
            delta = self.positions[b] - self.positions[a]
            dist = np.sqrt((delta**2).sum(axis=1)) + EPSILON
            rest = self.rest_lengths[group]
            # Each end takes half the correction, as in Toxiclibs (both inverse weights are
            # 1); a locked end doesn't move, and its half is simply not applied.
            error = (dist - rest) / (dist * 2) * self.strengths[group]
            error = np.where(self.min_distance[group] & (dist >= rest), 0.0, error)
            step = delta * error[:, None]
            self.positions[a] += step * ~self.locked[a][:, None]
            self.positions[b] -= step * ~self.locked[b][:, None]

    # --- inspection -------------------------------------------------------------------------
    def spring_lengths(self) -> FloatArray:
        a, b = self.spring_ends[:, 0], self.spring_ends[:, 1]
        return np.asarray(np.linalg.norm(self.positions[b] - self.positions[a], axis=1))
