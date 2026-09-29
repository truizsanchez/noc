"""Angular motion, oscillation, waves, springs and pendulums."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from noc.common.mathutils import clamp
from noc.common.noise import Noise
from noc.common.physics import Mover
from noc.common.vector import Vector

TWO_PI = 2 * math.pi


# --- angular motion ------------------------------------------------------------------------
@dataclass
class Spinner:
    """An angle with angular velocity and acceleration: ``angle += velocity += acceleration``.

    ``damping`` (Exercise 3.2) multiplies the velocity every frame.
    """

    angle: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    damping: float = 1.0

    def update(self) -> None:
        self.velocity = (self.velocity + self.acceleration) * self.damping
        self.angle += self.velocity


@dataclass
class RotatingMover:
    """Example 3.2: a mover that spins according to its horizontal acceleration."""

    body: Mover
    spin: Spinner = field(default_factory=Spinner)

    def update(self) -> None:
        # "Arbitrary" angular motion: faster sideways acceleration, faster spinning.
        self.spin.acceleration = self.body.acceleration.x / 10
        self.spin.velocity = clamp(self.spin.velocity + self.spin.acceleration, -0.1, 0.1)
        self.spin.angle += self.spin.velocity
        self.body.update()


@dataclass
class Spaceship:
    """Exercise 3.6: turns with the arrow keys and thrusts along its heading."""

    body: Mover
    heading: float = 0.0  # 0 points up, as in the solution
    damping: float = 0.995
    thrusting: bool = False

    def turn(self, angle: float) -> None:
        self.heading += angle

    def thrust(self, power: float = 0.1) -> None:
        self.body.apply_force(Vector.from_angle(self.heading - math.pi / 2) * power)
        self.thrusting = True

    def update(self, width: float, height: float, margin: float) -> None:
        body = self.body
        body.velocity = (body.velocity + body.acceleration) * self.damping
        body.acceleration = Vector()  # the damping replaces Mover.update's velocity step
        body.velocity = body.velocity.limit(body.top_speed)
        x, y = body.position + body.velocity
        if x > width + margin:
            x = -margin
        elif x < -margin:
            x = width + margin
        if y > height + margin:
            y = -margin
        elif y < -margin:
            y = height + margin
        body.position = Vector(x, y)


# --- oscillation ---------------------------------------------------------------------------
def simple_harmonic(amplitude: float, period: float, frame: float) -> float:
    """Example 3.5: ``amplitude * sin(2π * frame / period)``."""
    return amplitude * math.sin(TWO_PI * frame / period)


@dataclass
class Oscillator:
    """Example 3.7: independent oscillations along x and y."""

    angular_velocity: Vector
    amplitude: Vector
    angle: Vector = Vector()

    def update(self) -> None:
        self.angle += self.angular_velocity

    def offset(self) -> Vector:
        return Vector(
            math.sin(self.angle.x) * self.amplitude.x, math.sin(self.angle.y) * self.amplitude.y
        )


@dataclass
class Wave:
    """Exercise 3.11: a sine wave of ``width`` pixels, sampled every ``spacing`` pixels."""

    amplitude: float
    period: float  # pixels per cycle
    width: float
    spacing: float = 8.0
    theta: float = 0.0
    speed: float = 0.02

    def update(self) -> None:
        self.theta += self.speed

    def heights(self) -> list[float]:
        dx = TWO_PI / self.period * self.spacing
        count = int(self.width // self.spacing)
        return [math.sin(self.theta + i * dx) * self.amplitude for i in range(count)]


def additive_wave(waves: list[Wave], count: int) -> list[float]:
    """Exercise 3.12: several waves added together (alternating sine and cosine)."""
    total = [0.0] * count
    for j, wave in enumerate(waves):
        dx = TWO_PI / wave.period * wave.spacing
        trig = math.sin if j % 2 == 0 else math.cos
        for i in range(count):
            total[i] += trig(wave.theta + i * dx) * wave.amplitude
    return total


def noise_wave(noise: Noise, start: float, count: int, step: float = 0.1) -> list[float]:
    """Exercise 3.10: a "wave" whose heights come from Perlin noise instead of sine (0 to 1)."""
    return [noise(start + i * step) for i in range(count)]


# --- springs -------------------------------------------------------------------------------
@dataclass
class Spring:
    """Hooke's law: ``F = -k * stretch``, along the spring (Example 3.10)."""

    anchor: Vector
    rest_length: float
    k: float = 0.2

    def force(self, position: Vector) -> Vector:
        """The spring's force on whatever is attached at ``position``."""
        direction = position - self.anchor
        stretch = direction.mag() - self.rest_length
        return direction.set_mag(-self.k * stretch)

    def constrain(self, bob: Mover, shortest: float, longest: float) -> None:
        """Exercise 3.13: keep the spring's length within bounds, stopping the bob there."""
        direction = bob.position - self.anchor
        length = direction.mag()
        if not shortest <= length <= longest:
            bob.position = self.anchor + direction.set_mag(clamp(length, shortest, longest))
            bob.velocity = Vector()


def connect(a: Mover, b: Mover, rest_length: float, k: float = 0.2) -> None:
    """Exercise 3.14: a spring between two movable bobs pulls (or pushes) on both ends."""
    force = Spring(a.position, rest_length, k).force(b.position)
    b.apply_force(force)
    a.apply_force(-force)


# --- pendulums -----------------------------------------------------------------------------
@dataclass
class Pendulum:
    """Example 3.11: angular acceleration ``-g / r * sin(angle)``, with a little damping.

    The angle is measured from straight down; positive swings to the right.
    """

    pivot: Vector
    length: float
    angle: float = math.pi / 4
    velocity: float = 0.0
    gravity: float = 0.4
    damping: float = 0.995

    def update(self) -> None:
        self.velocity += -self.gravity / self.length * math.sin(self.angle)
        self.angle += self.velocity
        self.velocity *= self.damping

    @property
    def bob(self) -> Vector:
        """Polar to Cartesian, relative to the pivot."""
        return self.pivot + Vector(math.sin(self.angle), math.cos(self.angle)) * self.length

    def point_at(self, target: Vector) -> None:
        """Swing to face ``target`` (dragging the bob) and stop."""
        offset = target - self.pivot
        self.angle = math.atan2(offset.x, offset.y)
        self.velocity = 0.0


@dataclass
class DoublePendulum:
    """Exercise 3.15's solution: the exact equations of motion of a double pendulum."""

    r1: float = 100.0
    r2: float = 100.0
    m1: float = 10.0
    m2: float = 10.0
    a1: float = math.pi / 2
    a2: float = math.pi / 2
    v1: float = 0.0
    v2: float = 0.0
    g: float = 1.0

    def accelerations(self) -> tuple[float, float]:
        r1, r2, m1, m2, a1, a2, v1, v2, g = (
            self.r1, self.r2, self.m1, self.m2, self.a1, self.a2, self.v1, self.v2, self.g,
        )  # fmt: skip
        den = 2 * m1 + m2 - m2 * math.cos(2 * a1 - 2 * a2)
        acc1 = (
            -g * (2 * m1 + m2) * math.sin(a1)
            - m2 * g * math.sin(a1 - 2 * a2)
            - 2 * math.sin(a1 - a2) * m2 * (v2 * v2 * r2 + v1 * v1 * r1 * math.cos(a1 - a2))
        ) / (r1 * den)
        acc2 = (
            2
            * math.sin(a1 - a2)
            * (
                v1 * v1 * r1 * (m1 + m2)
                + g * (m1 + m2) * math.cos(a1)
                + v2 * v2 * r2 * m2 * math.cos(a1 - a2)
            )
        ) / (r2 * den)
        return acc1, acc2

    def update(self, substeps: int = 1) -> None:
        """Advance one frame, in ``substeps`` smaller Euler steps.

        The solution takes one step per frame, like every other sketch, but this system is
        chaotic and the error snowballs: the total energy swings by twice its scale within
        ten seconds. Fifty substeps keep the swings near a tenth of it.
        """
        dt = 1 / substeps
        for _ in range(substeps):
            acc1, acc2 = self.accelerations()
            self.v1 += acc1 * dt
            self.v2 += acc2 * dt
            self.a1 += self.v1 * dt
            self.a2 += self.v2 * dt

    def positions(self) -> tuple[Vector, Vector]:
        """Both bobs, relative to the pivot."""
        first = Vector(math.sin(self.a1), math.cos(self.a1)) * self.r1
        return first, first + Vector(math.sin(self.a2), math.cos(self.a2)) * self.r2

    def energy(self) -> float:
        """Kinetic plus potential energy (y down, so height is ``-y``)."""
        p1, p2 = self.positions()
        vel1 = Vector(math.cos(self.a1), -math.sin(self.a1)) * (self.r1 * self.v1)
        vel2 = vel1 + Vector(math.cos(self.a2), -math.sin(self.a2)) * (self.r2 * self.v2)
        kinetic = 0.5 * self.m1 * vel1.mag_sq() + 0.5 * self.m2 * vel2.mag_sq()
        potential = -self.g * (self.m1 * p1.y + self.m2 * p2.y)
        return kinetic + potential


# --- exercises with forces at an angle -----------------------------------------------------
@dataclass
class Incline:
    """Exercise 3.17: a box sliding down a slope, with friction proportional to the normal force.

    Motion is along the slope: gravity's component ``g * sin(θ)`` pulls it down, friction
    ``μ * g * cos(θ)`` resists while it moves (static friction holds it if gravity is weaker).
    """

    angle: float
    mu: float
    gravity: float = 0.1
    distance: float = 0.0  # along the slope, from the top
    speed: float = 0.0

    def update(self) -> None:
        pull = self.gravity * math.sin(self.angle)
        grip = self.mu * self.gravity * math.cos(self.angle)
        if self.speed == 0 and pull <= grip:
            return  # static friction wins
        self.speed = max(0.0, self.speed + pull - grip)
        self.distance += self.speed
