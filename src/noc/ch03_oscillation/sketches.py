"""Chapter 3's examples and exercises, in book order."""

from __future__ import annotations

import math
import random
from collections import deque

import arcade

from noc.ch03_oscillation.oscillation import (
    DoublePendulum,
    Incline,
    Oscillator,
    Pendulum,
    RotatingMover,
    Spaceship,
    Spinner,
    Spring,
    Wave,
    additive_wave,
    connect,
    noise_wave,
    simple_harmonic,
)
from noc.common.mathutils import remap
from noc.common.noise import Noise
from noc.common.physics import Mover, attraction, weight
from noc.common.vector import Vector
from noc.common.view import Canvas, Sketch, gray

GRAY = gray(127)
TRANSLUCENT = gray(127, 127)


def draw_baton(canvas: Canvas, angle: float, half: float) -> None:
    with canvas.pushed():
        canvas.translate(canvas.width / 2, canvas.height / 2)
        canvas.rotate(angle)
        canvas.line(-half, 0, half, 0, weight=2)
        canvas.circle(half, 0, 16, fill=GRAY, weight=2)
        canvas.circle(-half, 0, 16, fill=GRAY, weight=2)


# --- angular motion ------------------------------------------------------------------------
class Baton(Sketch):
    title = "Exercise 3.1: A Spinning Baton"

    def __init__(self) -> None:
        super().__init__()
        self.angle = 0.0

    def step(self) -> None:
        self.angle += 0.1

    def draw(self, canvas: Canvas) -> None:
        draw_baton(canvas, self.angle, 50)


class AngularMotion(Sketch):
    title = "Example 3.1: Angular Motion Using rotate()"

    def __init__(self) -> None:
        super().__init__()
        self.spin = Spinner(acceleration=0.0001)

    def step(self) -> None:
        self.spin.update()

    def draw(self, canvas: Canvas) -> None:
        draw_baton(canvas, self.spin.angle, 60)


class BatonWithDrag(Sketch):
    title = "Exercise 3.2: A Baton Spun by the Mouse, Slowed by Drag"
    help = ("Hold the mouse button: the farther right of center, the harder the spin",)

    def __init__(self) -> None:
        super().__init__()
        self.spin = Spinner(damping=0.98)

    def step(self) -> None:
        push = (self.mouse[0] - self.canvas.width / 2) / self.canvas.width
        self.spin.acceleration = push * 0.02 if self.mouse_pressed else 0.0
        self.spin.update()

    def draw(self, canvas: Canvas) -> None:
        draw_baton(canvas, self.spin.angle, 60)
        self.status = f"angular velocity {self.spin.velocity:+.3f}"


class ForcesWithAngularMotion(Sketch):
    title = "Example 3.2: Forces with (Arbitrary) Angular Motion"

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.attractor = Mover(Vector(width / 2, height / 2), mass=20)
        self.movers = [
            RotatingMover(
                Mover(
                    Vector(random.uniform(0, width), random.uniform(0, height)),
                    Vector(random.uniform(-1, 1), random.uniform(-1, 1)),
                    random.uniform(0.1, 2),
                )
            )
            for _ in range(20)
        ]

    def step(self) -> None:
        a = self.attractor
        for mover in self.movers:
            body = mover.body
            body.apply_force(attraction(a.position, a.mass, body.position, body.mass))
            mover.update()

    def draw(self, canvas: Canvas) -> None:
        canvas.circle(*self.attractor.position.xy, 40, fill=gray(175, 200))
        for mover in self.movers:
            radius = mover.body.mass * 8
            with canvas.pushed():
                canvas.translate(*mover.body.position.xy)
                canvas.rotate(mover.spin.angle)
                canvas.circle(0, 0, radius * 2, fill=TRANSLUCENT, weight=2)
                canvas.line(0, 0, radius, 0, weight=2)


class Cannon(Sketch):
    title = "Exercise 3.3: A Cannon"
    help = ("Space: fire   Up/Down: aim",)

    def __init__(self) -> None:
        super().__init__()
        self.aim = -math.pi / 4
        self.shells: list[RotatingMover] = []

    @property
    def muzzle(self) -> Vector:
        return Vector(40, self.canvas.height - 20) + Vector.from_angle(self.aim, 40)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        match symbol:
            case arcade.key.SPACE:
                shell = RotatingMover(Mover(self.muzzle, mass=2))
                shell.body.apply_force(Vector.from_angle(self.aim, 18))  # the blast: once
                self.shells.append(shell)
            case arcade.key.UP:
                self.aim = max(self.aim - 0.1, -math.pi / 2)
            case arcade.key.DOWN:
                self.aim = min(self.aim + 0.1, 0)
            case _:
                return super().on_key_press(symbol, modifiers)
        return True

    def step(self) -> None:
        for shell in self.shells:
            shell.body.apply_force(weight(shell.body.mass, Vector(0, 0.2)))  # always
            shell.update()
        self.shells = [s for s in self.shells if s.body.position.y < self.canvas.height + 20]

    def draw(self, canvas: Canvas) -> None:
        with canvas.pushed():
            canvas.translate(40, canvas.height - 20)
            canvas.rotate(self.aim)
            canvas.rect(0, -8, 45, 16, fill=gray(80), weight=2)
        canvas.circle(40, canvas.height - 20, 30, fill=gray(60), weight=2)
        for shell in self.shells:
            with canvas.pushed():
                canvas.translate(*shell.body.position.xy)
                canvas.rotate(shell.spin.angle)
                canvas.rect(0, 0, 16, 8, fill=GRAY, weight=2, center=True)


class PointingInDirectionOfMotion(Sketch):
    title = "Example 3.3: Pointing in the Direction of Motion"
    help = ("Move the mouse",)

    def __init__(self) -> None:
        super().__init__()
        self.mover = Mover(Vector(self.canvas.width / 2, self.canvas.height / 2), top_speed=4)

    def step(self) -> None:
        self.mover.apply_force((Vector(*self.mouse) - self.mover.position).set_mag(0.5))
        self.mover.update()

    def draw(self, canvas: Canvas) -> None:
        draw_pointing_rect(canvas, self.mover)


def draw_pointing_rect(canvas: Canvas, mover: Mover) -> None:
    with canvas.pushed():
        canvas.translate(*mover.position.xy)
        canvas.rotate(mover.velocity.heading())
        canvas.rect(0, 0, 30, 10, fill=GRAY, weight=2, center=True)


class DrivingCar(Sketch):
    title = "Exercise 3.4: A Car Driven with the Arrow Keys"
    help = ("Arrow keys: accelerate in that direction",)

    def __init__(self) -> None:
        super().__init__()
        self.car = Mover(Vector(self.canvas.width / 2, self.canvas.height / 2), top_speed=5)

    def step(self) -> None:
        directions = {
            arcade.key.LEFT: Vector(-1, 0),
            arcade.key.RIGHT: Vector(1, 0),
            arcade.key.UP: Vector(0, -1),
            arcade.key.DOWN: Vector(0, 1),
        }
        for key, direction in directions.items():
            if key in self.held:
                self.car.apply_force(direction * 0.2)
        self.car.velocity *= 0.98  # rolling resistance
        self.car.update()
        x, y = self.car.position
        self.car.position = Vector(x % self.canvas.width, y % self.canvas.height)

    def draw(self, canvas: Canvas) -> None:
        draw_pointing_rect(canvas, self.car)


class PolarToCartesian(Sketch):
    title = "Example 3.4: Polar to Cartesian"

    def __init__(self) -> None:
        super().__init__()
        self.r, self.theta = self.canvas.height * 0.45, 0.0

    def step(self) -> None:
        self.theta += 0.02

    def draw(self, canvas: Canvas) -> None:
        position = Vector.from_angle(self.theta, self.r)
        canvas.translate(canvas.width / 2, canvas.height / 2)
        canvas.line(0, 0, *position.xy, weight=2)
        canvas.circle(*position.xy, 48, fill=GRAY, weight=2)


class Spiral(Sketch):
    title = "Exercise 3.5: A Spiral"
    background = None

    def __init__(self) -> None:
        super().__init__()
        self.r, self.theta = 0.0, 0.0

    def step(self) -> None:
        self.theta += 0.01
        self.r += 0.05

    def draw(self, canvas: Canvas) -> None:
        x, y = Vector.from_angle(self.theta, self.r)
        canvas.circle(x + canvas.width / 2, y + canvas.height / 2, 16, fill=gray(0), stroke=None)


class Asteroids(Sketch):
    title = "Exercise 3.6: The Asteroids Spaceship"
    help = ("Left/Right: turn   Z: thrust",)
    ship_size = 16

    def __init__(self) -> None:
        super().__init__()
        center = Vector(self.canvas.width / 2, self.canvas.height / 2)
        self.ship = Spaceship(Mover(center, top_speed=6))

    def step(self) -> None:
        if arcade.key.LEFT in self.held:
            self.ship.turn(-0.03)
        elif arcade.key.RIGHT in self.held:
            self.ship.turn(0.03)
        self.ship.thrusting = arcade.key.Z in self.held
        if self.ship.thrusting:
            self.ship.thrust()
        self.ship.update(self.canvas.width, self.canvas.height, self.ship_size * 2)

    def draw(self, canvas: Canvas) -> None:
        r = self.ship_size
        canvas.translate(*self.ship.body.position.xy)
        canvas.rotate(self.ship.heading)
        flame = (255, 0, 0) if self.ship.thrusting else gray(175)
        canvas.rect(-r / 2, r, r / 3, r / 2, fill=flame, weight=2, center=True)
        canvas.rect(r / 2, r, r / 3, r / 2, fill=flame, weight=2, center=True)
        canvas.polygon([(-r, r), (0, -r), (r, r)], fill=gray(175), weight=2)


# --- oscillation ---------------------------------------------------------------------------
def draw_pendulum_arm(canvas: Canvas, x: float, y: float, diameter: float = 48) -> None:
    canvas.line(0, 0, x, y, weight=2)
    canvas.circle(x, y, diameter, fill=GRAY, weight=2)


class SimpleHarmonicMotion(Sketch):
    title = "Example 3.5: Simple Harmonic Motion I"

    def draw(self, canvas: Canvas) -> None:
        x = simple_harmonic(200, 120, self.frame_count)
        canvas.translate(canvas.width / 2, canvas.height / 2)
        draw_pendulum_arm(canvas, x, 0)


class SineBob(Sketch):
    title = "Exercise 3.7: A Bob Hanging from a Spring, with Sine"

    def draw(self, canvas: Canvas) -> None:
        y = remap(math.sin(self.frame_count * 0.05), -1, 1, 60, 200)
        canvas.translate(canvas.width / 2, 0)
        draw_pendulum_arm(canvas, 0, y)


class SimpleHarmonicMotionII(Sketch):
    title = "Example 3.6: Simple Harmonic Motion II"

    def __init__(self) -> None:
        super().__init__()
        self.angle = 0.0

    def step(self) -> None:
        self.angle += 0.05

    def draw(self, canvas: Canvas) -> None:
        canvas.translate(canvas.width / 2, canvas.height / 2)
        draw_pendulum_arm(canvas, 200 * math.sin(self.angle), 0)


class OscillatorObjects(Sketch):
    title = "Example 3.7: Oscillator Objects"

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.oscillators = [
            Oscillator(
                Vector(random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)),
                Vector(random.uniform(20, width / 2), random.uniform(20, height / 2)),
            )
            for _ in range(10)
        ]

    def step(self) -> None:
        for oscillator in self.oscillators:
            oscillator.update()

    def draw(self, canvas: Canvas) -> None:
        canvas.translate(canvas.width / 2, canvas.height / 2)
        for oscillator in self.oscillators:
            draw_pendulum_arm(canvas, *oscillator.offset().xy, 32)


# --- waves ---------------------------------------------------------------------------------
def draw_wave(canvas: Canvas, heights: list[float], spacing: float, origin: Vector) -> None:
    for i, y in enumerate(heights):
        canvas.circle(origin.x + i * spacing, origin.y + y, 48, fill=TRANSLUCENT, weight=2)


class StaticWave(Sketch):
    title = "Example 3.8: Static Wave"

    def draw(self, canvas: Canvas) -> None:
        heights = [100 * math.sin(i * 0.2) for i in range(int(canvas.width // 24) + 1)]
        draw_wave(canvas, heights, 24, Vector(0, canvas.height / 2))


class TheWave(Sketch):
    title = "Example 3.9: The Wave"

    def __init__(self) -> None:
        super().__init__()
        self.start_angle = 0.0

    def step(self) -> None:
        self.start_angle += 0.02

    def draw(self, canvas: Canvas) -> None:
        count = int(canvas.width // 24) + 1
        heights = [
            remap(math.sin(self.start_angle + i * 0.2), -1, 1, 0, canvas.height)
            for i in range(count)
        ]
        draw_wave(canvas, heights, 24, Vector())


class NoiseWave(Sketch):
    title = "Exercise 3.10: A Wave from Perlin Noise"

    def __init__(self) -> None:
        super().__init__()
        self.noise, self.start = Noise(), 0.0

    def step(self) -> None:
        self.start += 0.02

    def draw(self, canvas: Canvas) -> None:
        count = int(canvas.width // 24) + 1
        heights = [h * canvas.height for h in noise_wave(self.noise, self.start, count)]
        draw_wave(canvas, heights, 24, Vector())


class TwoWaves(Sketch):
    title = "Exercise 3.11: A Wave Class, Two Waves"

    def __init__(self) -> None:
        super().__init__()
        self.waves = [(Wave(20, 600, 100), Vector(50, 75)), (Wave(40, 180, 300), Vector(300, 120))]

    def step(self) -> None:
        for wave, _ in self.waves:
            wave.update()

    def draw(self, canvas: Canvas) -> None:
        for wave, origin in self.waves:
            for i, y in enumerate(wave.heights()):
                canvas.circle(origin.x + i * wave.spacing, origin.y + y, 48, fill=gray(0, 50))


class AdditiveWave(Sketch):
    title = "Exercise 3.12: Additive Waves"

    def __init__(self) -> None:
        super().__init__()
        self.waves = [Wave(random.uniform(10, 30), random.uniform(100, 300), 0) for _ in range(5)]

    def step(self) -> None:
        for wave in self.waves:
            wave.update()

    def draw(self, canvas: Canvas) -> None:
        count = int((canvas.width + 16) // 8)
        for i, y in enumerate(additive_wave(self.waves, count)):
            canvas.circle(i * 8, canvas.height / 2 + y, 32, fill=gray(0, 100))


# --- springs and pendulums -----------------------------------------------------------------
class Grabbable:
    """Mouse dragging for a bob: grab within its radius, follow the mouse, let go."""

    def __init__(self) -> None:
        self.grab_offset: Vector | None = None

    def grab(self, mouse: Vector, position: Vector, radius: float) -> None:
        if mouse.dist(position) < radius:
            self.grab_offset = position - mouse

    def release(self) -> None:
        self.grab_offset = None


class SpringConnection(Sketch):
    title = "Example 3.10: A Spring Connection"
    help = ("Drag the bob",)

    def __init__(self) -> None:
        super().__init__()
        width = self.canvas.width
        self.spring = Spring(Vector(width / 2, 10), 100)
        self.bob = Mover(Vector(width / 2, 100), mass=24)
        self.grab = Grabbable()

    def mouse_down(self) -> None:
        self.grab.grab(Vector(*self.mouse), self.bob.position, self.bob.mass)

    def mouse_up(self) -> None:
        self.grab.release()

    def step(self) -> None:
        bob = self.bob
        bob.apply_force(Vector(0, 2))
        bob.velocity *= 0.98  # damping
        bob.update()
        if self.grab.grab_offset is not None:
            bob.position = Vector(*self.mouse) + self.grab.grab_offset
        bob.apply_force(self.spring.force(bob.position))
        self.spring.constrain(bob, 30, 200)

    def draw(self, canvas: Canvas) -> None:
        anchor, bob = self.spring.anchor, self.bob
        canvas.line(*bob.position.xy, *anchor.xy)
        fill = gray(200) if self.grab.grab_offset is not None else GRAY
        canvas.circle(*bob.position.xy, bob.mass * 2, fill=fill, weight=2)
        canvas.circle(*anchor.xy, 10, fill=GRAY)


class SpringChain(Sketch):
    title = "Exercise 3.14: Bobs Connected by Springs"
    help = ("Drag any bob",)

    def __init__(self) -> None:
        super().__init__()
        self.anchor = Vector(self.canvas.width / 2, 10)
        self.bobs = [
            Mover(Vector(self.canvas.width / 2 + i * 30, 10 + i * 40), mass=8) for i in range(1, 5)
        ]
        self.grabs = [Grabbable() for _ in self.bobs]

    def mouse_down(self) -> None:
        for bob, grab in zip(self.bobs, self.grabs, strict=True):
            grab.grab(Vector(*self.mouse), bob.position, 12)

    def mouse_up(self) -> None:
        for grab in self.grabs:
            grab.release()

    def step(self) -> None:
        first = self.bobs[0]
        first.apply_force(Spring(self.anchor, 30).force(first.position))
        for a, b in zip(self.bobs, self.bobs[1:], strict=False):
            connect(a, b, 30)
        for bob, grab in zip(self.bobs, self.grabs, strict=True):
            bob.apply_force(weight(bob.mass, Vector(0, 0.1)))
            bob.velocity *= 0.98
            bob.update()
            if grab.grab_offset is not None:
                bob.position = Vector(*self.mouse) + grab.grab_offset
                bob.velocity = Vector()

    def draw(self, canvas: Canvas) -> None:
        points = [self.anchor.xy] + [bob.position.xy for bob in self.bobs]
        canvas.polyline(points)
        canvas.circle(*self.anchor.xy, 10, fill=GRAY)
        for bob in self.bobs:
            canvas.circle(*bob.position.xy, 24, fill=GRAY, weight=2)


class SwingingPendulum(Sketch):
    title = "Example 3.11: Swinging Pendulum"
    help = ("Drag the bob",)

    def __init__(self) -> None:
        super().__init__()
        self.pendulum = Pendulum(Vector(self.canvas.width / 2, 0), 175)
        self.dragging = False

    def mouse_down(self) -> None:
        self.dragging = Vector(*self.mouse).dist(self.pendulum.bob) < 24

    def mouse_up(self) -> None:
        self.dragging = False

    def step(self) -> None:
        if self.dragging:
            self.pendulum.point_at(Vector(*self.mouse))
        else:
            self.pendulum.update()

    def draw(self, canvas: Canvas) -> None:
        pivot, bob = self.pendulum.pivot, self.pendulum.bob
        canvas.line(*pivot.xy, *bob.xy, weight=2)
        canvas.circle(*bob.xy, 48, fill=GRAY, weight=2)


class DoublePendulumSketch(Sketch):
    title = "Exercise 3.15: A Double Pendulum"
    trail_length = 3000

    def __init__(self) -> None:
        super().__init__()
        self.pendulum = DoublePendulum()
        self.trail: deque[tuple[float, float]] = deque(maxlen=self.trail_length)

    def step(self) -> None:
        self.pendulum.update(substeps=50)
        self.trail.append(self.pendulum.positions()[1].xy)

    def draw(self, canvas: Canvas) -> None:
        canvas.translate(canvas.width / 2, 20)
        if len(self.trail) > 1:
            canvas.polyline(list(self.trail), gray(0, 120))
        first, second = self.pendulum.positions()
        canvas.line(0, 0, *first.xy, weight=2)
        canvas.line(*first.xy, *second.xy, weight=2)
        canvas.circle(*first.xy, 10, fill=gray(0))
        canvas.circle(*second.xy, 10, fill=gray(0))
        self.status = f"energy {self.pendulum.energy():.1f}"


class SlidingBox(Sketch):
    title = "Exercise 3.17: A Box Sliding Down an Incline with Friction"
    help = ("Up/Down: steeper/flatter   Left/Right: less/more friction",)

    def __init__(self) -> None:
        super().__init__()
        self.incline = Incline(math.radians(25), 0.3)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        incline = self.incline
        match symbol:
            case arcade.key.UP | arcade.key.DOWN:
                delta = math.radians(5 if symbol == arcade.key.UP else -5)
                incline.angle = min(max(incline.angle + delta, math.radians(5)), math.radians(45))
            case arcade.key.LEFT | arcade.key.RIGHT:
                delta = 0.05 if symbol == arcade.key.RIGHT else -0.05
                incline.mu = round(min(max(incline.mu + delta, 0.0), 1.0), 2)
            case _:
                return super().on_key_press(symbol, modifiers)
        incline.distance, incline.speed = 0.0, 0.0
        return True

    def step(self) -> None:
        self.incline.update()
        if self.incline.distance > self.slope_length():
            self.incline.distance, self.incline.speed = 0.0, 0.0

    def slope_length(self) -> float:
        return (self.canvas.height - 40) / math.sin(self.incline.angle)

    def draw(self, canvas: Canvas) -> None:
        incline = self.incline
        top = Vector(40, 20)
        bottom = top + Vector.from_angle(incline.angle, self.slope_length())
        canvas.polygon([top.xy, bottom.xy, (top.x, bottom.y)], fill=gray(220))
        with canvas.pushed():
            canvas.translate(*top.xy)
            canvas.rotate(incline.angle)
            canvas.rect(incline.distance, -24, 40, 24, fill=GRAY, weight=2)
        self.status = (
            f"angle {math.degrees(incline.angle):.0f}°  μ {incline.mu:.2f}  "
            f"tan(angle) {math.tan(incline.angle):.2f}"
        )


SKETCHES: tuple[type[Sketch], ...] = (
    Baton,
    AngularMotion,
    BatonWithDrag,
    ForcesWithAngularMotion,
    Cannon,
    PointingInDirectionOfMotion,
    DrivingCar,
    PolarToCartesian,
    Spiral,
    Asteroids,
    SimpleHarmonicMotion,
    SineBob,
    SimpleHarmonicMotionII,
    OscillatorObjects,
    StaticWave,
    TheWave,
    NoiseWave,
    TwoWaves,
    AdditiveWave,
    SpringConnection,
    SpringChain,
    SwingingPendulum,
    DoublePendulumSketch,
    SlidingBox,
)
