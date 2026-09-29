"""Vehicles and Reynolds's steering behaviors: *steering = desired velocity - velocity*.

Each behavior is a function that returns a steering force (already limited to the vehicle's
``max_force``) instead of applying it, so behaviors can be weighted and combined
(Example 5.10) before ``vehicle.apply_force``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from noc.common.mathutils import clamp, remap
from noc.common.noise import Noise
from noc.common.physics import Mover
from noc.common.vector import Vector

type FloatArray = npt.NDArray[np.float64]


@dataclass
class Vehicle(Mover):
    """A mover with a top speed and a limit on how hard it can steer."""

    max_speed: float = 4.0
    max_force: float = 0.1
    r: float = 6.0

    def __post_init__(self) -> None:
        self.top_speed = self.max_speed

    def steer_toward(self, desired: Vector) -> Vector:
        """Reynolds's formula: the force that turns the velocity into ``desired``."""
        return (desired - self.velocity).limit(self.max_force)

    def wrap(self, width: float, height: float) -> None:
        """Leave one edge, come back on the other, ``r`` beyond the canvas."""
        x, y, r = self.position.x, self.position.y, self.r
        if x < -r:
            x = width + r
        elif x > width + r:
            x = -r
        if y < -r:
            y = height + r
        elif y > height + r:
            y = -r
        self.position = Vector(x, y)


# --- seeking and fleeing -------------------------------------------------------------------
def seek(vehicle: Vehicle, target: Vector) -> Vector:
    """Example 5.1: full speed straight at the target."""
    return vehicle.steer_toward((target - vehicle.position).set_mag(vehicle.max_speed))


def flee(vehicle: Vehicle, target: Vector) -> Vector:
    """Exercise 5.1: full speed straight away from the target."""
    return vehicle.steer_toward((vehicle.position - target).set_mag(vehicle.max_speed))


def pursue(vehicle: Vehicle, quarry: Mover, lookahead: float = 10) -> Vector:
    """Exercise 5.3: seek where the quarry will be ``lookahead`` frames from now."""
    return seek(vehicle, predicted(quarry, lookahead))


def evade(vehicle: Vehicle, pursuer: Mover, lookahead: float = 10) -> Vector:
    """Exercise 5.3: flee from where the pursuer will be."""
    return flee(vehicle, predicted(pursuer, lookahead))


def predicted(mover: Mover, frames: float) -> Vector:
    return mover.position + mover.velocity * frames


def arrive(vehicle: Vehicle, target: Vector, slowing_radius: float = 100) -> Vector:
    """Example 5.2: like seek, but slowing down linearly inside ``slowing_radius``."""
    desired = target - vehicle.position
    distance = desired.mag()
    speed = vehicle.max_speed
    if distance < slowing_radius:
        speed = remap(distance, 0, slowing_radius, 0, vehicle.max_speed)
    return vehicle.steer_toward(desired.set_mag(speed))


@dataclass
class Wander:
    """Exercise 5.4: seek a point that drifts around a circle projected ahead of the vehicle."""

    radius: float = 25.0
    distance: float = 80.0
    change: float = 0.3
    theta: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    def target(self, vehicle: Vehicle) -> tuple[Vector, Vector]:
        """``(circle center, target on the circle)``, for drawing."""
        self.theta += self.rng.uniform(-self.change, self.change)
        center = vehicle.position + vehicle.velocity.set_mag(self.distance)
        angle = self.theta + vehicle.velocity.heading()
        return center, center + Vector.from_angle(angle, self.radius)

    def __call__(self, vehicle: Vehicle) -> Vector:
        return seek(vehicle, self.target(vehicle)[1])


def stay_within(vehicle: Vehicle, width: float, height: float, offset: float) -> Vector:
    """Example 5.3: near a wall, desire full speed away from it, keeping the other axis."""
    x, y = vehicle.position
    vx, vy = vehicle.velocity
    desired: Vector | None = None
    if x < offset:
        desired = Vector(vehicle.max_speed, vy)
    elif x > width - offset:
        desired = Vector(-vehicle.max_speed, vy)
    if y < offset:
        desired = Vector(vx, vehicle.max_speed)
    elif y > height - offset:
        desired = Vector(vx, -vehicle.max_speed)
    if desired is None:
        return Vector()
    return vehicle.steer_toward(desired.set_mag(vehicle.max_speed))


# --- flow fields ---------------------------------------------------------------------------
@dataclass
class FlowField:
    """Example 5.4: a grid of directions, one per ``resolution x resolution`` cell.

    ``angles[col, row]`` holds the direction of each cell, in radians.
    """

    resolution: int
    angles: FloatArray

    @property
    def cols(self) -> int:
        return int(self.angles.shape[0])

    @property
    def rows(self) -> int:
        return int(self.angles.shape[1])

    @classmethod
    def from_noise(
        cls, width: int, height: int, resolution: int, noise: Noise, z: float = 0.0
    ) -> FlowField:
        cols, rows = width // resolution, height // resolution
        xs, ys = np.meshgrid(np.arange(cols) * 0.1, np.arange(rows) * 0.1, indexing="ij")
        return cls(resolution, noise.grid(xs, ys, z) * 2 * math.pi)

    @classmethod
    def swirl(cls, width: int, height: int, resolution: int) -> FlowField:
        """Exercise 5.6: every vector turns around the center of the canvas (clockwise)."""
        cols, rows = width // resolution, height // resolution
        i, j = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")
        x = (i + 0.5) * resolution - width / 2
        y = (j + 0.5) * resolution - height / 2
        return cls(resolution, np.arctan2(y, x) + math.pi / 2)

    def lookup(self, position: Vector) -> Vector:
        col = int(clamp(position.x // self.resolution, 0, self.cols - 1))
        row = int(clamp(position.y // self.resolution, 0, self.rows - 1))
        return Vector.from_angle(float(self.angles[col, row]))


def follow_field(vehicle: Vehicle, flow: FlowField) -> Vector:
    """Desire the field's direction at full speed."""
    return vehicle.steer_toward(flow.lookup(vehicle.position) * vehicle.max_speed)


# --- paths ---------------------------------------------------------------------------------
@dataclass
class Path:
    """Examples 5.5-5.8: a polyline with a radius; ``closed`` joins the last point to the first."""

    points: list[Vector]
    radius: float = 20.0
    closed: bool = False

    def segments(self) -> list[tuple[Vector, Vector]]:
        ends = [*self.points[1:], self.points[0]] if self.closed else self.points[1:]
        return list(zip(self.points, ends, strict=False))


def normal_point(p: Vector, a: Vector, b: Vector) -> Vector:
    """The point on the (infinite) line ``ab`` closest to ``p``: a scalar projection."""
    direction = (b - a).normalize()
    return a + direction * (p - a).dot(direction)


def closest_on_segment(p: Vector, a: Vector, b: Vector) -> Vector:
    """Like :func:`normal_point`, but kept between ``a`` and ``b``.

    The book tests ``normalPoint.x`` against the segment's ends, which works only for paths
    that run left to right; Exercise 5.10 asks for a general test, and clamping the projection
    to the segment is the simplest one.
    """
    ab = b - a
    t = clamp((p - a).dot(ab) / ab.mag_sq(), 0, 1)
    return a + ab * t


@dataclass(frozen=True)
class PathFollowing:
    """What :func:`follow_path` computed, for the debug drawing."""

    force: Vector
    future: Vector
    normal: Vector
    target: Vector


def follow_path(
    vehicle: Vehicle, path: Path, lookahead: float = 50, target_ahead: float = 10
) -> PathFollowing:
    """Example 5.8: predict the position, find the nearest point on the path, and if the
    prediction is outside the path's radius, seek a point a little farther along."""
    future = vehicle.position + vehicle.velocity.set_mag(lookahead)
    best: tuple[float, Vector, Vector] | None = None
    for a, b in path.segments():
        normal = closest_on_segment(future, a, b)
        distance = future.dist(normal)
        if best is None or distance < best[0]:
            best = (distance, normal, normal + (b - a).set_mag(target_ahead))
    assert best is not None, "a path needs at least two points"
    distance, normal, target = best
    force = seek(vehicle, target) if distance > path.radius else Vector()
    return PathFollowing(force, future, normal, target)


# --- group behaviors -----------------------------------------------------------------------
def separate(vehicle: Vehicle, others: Sequence[Vehicle], distance: float) -> Vector:
    """Example 5.9: steer away from neighbors closer than ``distance``, harder from closer ones.

    Each neighbor contributes a vector pointing away from it with magnitude ``1 / d``.
    """
    total = Vector()
    count = 0
    for other in others:
        d = vehicle.position.dist(other.position)
        if other is not vehicle and 0 < d < distance:
            total += (vehicle.position - other.position).set_mag(1 / d)
            count += 1
    if count == 0:
        return Vector()
    return vehicle.steer_toward(total.set_mag(vehicle.max_speed))


def align(vehicle: Vehicle, others: Sequence[Vehicle], distance: float = 50) -> Vector:
    """Example 5.11: steer toward the neighbors' average heading."""
    total = Vector()
    count = 0
    for other in others:
        d = vehicle.position.dist(other.position)
        if other is not vehicle and 0 < d < distance:
            total += other.velocity
            count += 1
    if count == 0:
        return Vector()
    return vehicle.steer_toward(total.set_mag(vehicle.max_speed))


def cohere(vehicle: Vehicle, others: Sequence[Vehicle], distance: float = 50) -> Vector:
    """Exercise 5.12 / Example 5.11: seek the neighbors' average position."""
    total = Vector()
    count = 0
    for other in others:
        d = vehicle.position.dist(other.position)
        if other is not vehicle and 0 < d < distance:
            total += other.position
            count += 1
    if count == 0:
        return Vector()
    return seek(vehicle, total / count)


@dataclass
class FlockWeights:
    separation: float = 1.5
    alignment: float = 1.0
    cohesion: float = 1.0
    separation_distance: float = 25.0
    neighbor_distance: float = 50.0


def flock(vehicle: Vehicle, others: Sequence[Vehicle], weights: FlockWeights) -> Vector:
    """Example 5.11: separation, alignment and cohesion, weighted and added.

    The same as calling :func:`separate`, :func:`align` and :func:`cohere`, but in one pass
    over the neighbors, measuring each distance once instead of three times.
    """
    px, py = vehicle.position
    away_x = away_y = heading_x = heading_y = center_x = center_y = 0.0
    near = neighbors = 0
    for other in others:
        dx, dy = px - other.position.x, py - other.position.y
        d = math.hypot(dx, dy)
        if d == 0 or other is vehicle:
            continue
        if d < weights.separation_distance:
            away_x += dx / (d * d)  # a unit vector away, divided by d
            away_y += dy / (d * d)
            near += 1
        if d < weights.neighbor_distance:
            heading_x += other.velocity.x
            heading_y += other.velocity.y
            center_x += other.position.x
            center_y += other.position.y
            neighbors += 1
    force = Vector()
    speed = vehicle.max_speed
    if near:
        force += vehicle.steer_toward(Vector(away_x, away_y).set_mag(speed)) * weights.separation
    if neighbors:
        heading = Vector(heading_x, heading_y).set_mag(speed)
        center = Vector(center_x / neighbors, center_y / neighbors)
        force += vehicle.steer_toward(heading) * weights.alignment
        force += seek(vehicle, center) * weights.cohesion
    return force
