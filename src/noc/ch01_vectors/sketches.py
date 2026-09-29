"""Chapter 1's examples and exercises, in book order."""

from __future__ import annotations

import random

import arcade

from noc.ch01_vectors.motion import (
    Mover,
    NoiseAcceleration,
    Train,
    random_acceleration,
    toward,
    toward_by_distance,
)
from noc.common.mathutils import clamp
from noc.common.noise import Noise
from noc.common.vector import Vector
from noc.common.view import Canvas, Sketch, gray


def draw_ball(canvas: Canvas, position: Vector) -> None:
    canvas.circle(*position.xy, 48, fill=gray(127), weight=2)


class BouncingBallNoVectors(Sketch):
    title = "Example 1.1: Bouncing Ball with No Vectors"

    def __init__(self) -> None:
        super().__init__()
        self.x, self.y = 100.0, 100.0
        self.xspeed, self.yspeed = 2.5, 2.0

    def step(self) -> None:
        self.x += self.xspeed
        self.y += self.yspeed
        if not 0 <= self.x <= self.canvas.width:
            self.xspeed *= -1
        if not 0 <= self.y <= self.canvas.height:
            self.yspeed *= -1

    def draw(self, canvas: Canvas) -> None:
        canvas.circle(self.x, self.y, 48, fill=gray(127), weight=2)


class BouncingBallWithVectors(Sketch):
    title = "Example 1.2: Bouncing Ball with Vectors!"

    def __init__(self) -> None:
        super().__init__()
        self.ball = Mover(Vector(100, 100), Vector(2.5, 2))

    def step(self) -> None:
        self.ball.update()
        self.ball.bounce_edges(self.canvas.width, self.canvas.height)

    def draw(self, canvas: Canvas) -> None:
        draw_ball(canvas, self.ball.position)


class WalkerWithVectors(Sketch):
    title = "Exercise 1.1: A Random Walker with Vectors"
    background = None

    def __init__(self) -> None:
        super().__init__()
        self.position = Vector(self.canvas.width / 2, self.canvas.height / 2)

    def step(self) -> None:
        x, y = self.position + Vector.random2d() * 2
        self.position = Vector(
            clamp(x, 0, self.canvas.width - 1), clamp(y, 0, self.canvas.height - 1)
        )

    def draw(self, canvas: Canvas) -> None:
        canvas.point(*self.position.xy, weight=2)


# --- vector math with the mouse --------------------------------------------------------
class MouseVectorSketch(Sketch):
    """The mouse position as a vector from the center of the canvas."""

    help = ("Move the mouse",)

    def mouse_from_center(self) -> Vector:
        return Vector(*self.mouse) - self.canvas_center()

    def canvas_center(self) -> Vector:
        return Vector(self.canvas.width / 2, self.canvas.height / 2)


class VectorSubtraction(MouseVectorSketch):
    title = "Example 1.3: Vector Subtraction"

    def draw(self, canvas: Canvas) -> None:
        mouse, center = Vector(*self.mouse), self.canvas_center()
        canvas.line(0, 0, *mouse.xy, gray(200), 4)
        canvas.line(0, 0, *center.xy, gray(200), 4)
        canvas.translate(*center.xy)
        canvas.line(0, 0, *(mouse - center).xy, weight=4)


class VectorMultiplication(MouseVectorSketch):
    title = "Example 1.4: Multiplying a Vector"

    def draw(self, canvas: Canvas) -> None:
        mouse = self.mouse_from_center()
        canvas.translate(*self.canvas_center().xy)
        canvas.line(0, 0, *mouse.xy, gray(200), 2)
        canvas.line(0, 0, *(mouse * 0.5).xy, weight=4)


class VectorMagnitude(MouseVectorSketch):
    title = "Example 1.5: Vector Magnitude"

    def draw(self, canvas: Canvas) -> None:
        mouse = self.mouse_from_center()
        canvas.rect(10, 10, mouse.mag(), 10, fill=gray(0))
        canvas.translate(*self.canvas_center().xy)
        canvas.line(0, 0, *mouse.xy)


class VectorNormalize(MouseVectorSketch):
    title = "Example 1.6: Normalizing a Vector"

    def draw(self, canvas: Canvas) -> None:
        mouse = self.mouse_from_center()
        canvas.translate(*self.canvas_center().xy)
        canvas.line(0, 0, *mouse.xy, gray(200), 2)
        canvas.line(0, 0, *(mouse.normalize() * 50).xy, weight=8)


# --- motion -----------------------------------------------------------------------------
class MoverSketch(Sketch):
    """A ball that leaves one edge of the canvas and comes back on the other."""

    def __init__(self) -> None:
        super().__init__()
        self.mover = self.make_mover()

    def make_mover(self) -> Mover:
        return Mover(Vector(self.canvas.width / 2, self.canvas.height / 2))

    def accelerate(self) -> None:
        """Set ``self.mover.acceleration`` for this frame (the part that varies)."""

    def step(self) -> None:
        self.accelerate()
        self.mover.update()
        self.mover.wrap_edges(self.canvas.width, self.canvas.height)

    def draw(self, canvas: Canvas) -> None:
        draw_ball(canvas, self.mover.position)


class MotionVelocity(MoverSketch):
    title = "Example 1.7: Motion 101 (Velocity)"

    def make_mover(self) -> Mover:
        width, height = self.canvas_size
        position = Vector(random.uniform(0, width), random.uniform(0, height))
        return Mover(position, Vector(random.uniform(-2, 2), random.uniform(-2, 2)))


class MotionConstantAcceleration(MoverSketch):
    title = "Example 1.8: Motion 101 (Velocity and Constant Acceleration)"

    def make_mover(self) -> Mover:
        mover = super().make_mover()
        mover.acceleration, mover.top_speed = Vector(-0.001, 0.01), 10
        return mover


class AccelerateAndBrake(Sketch):
    title = "Exercise 1.5: Accelerate and Brake"
    help = ("Up: accelerate   Down: brake",)
    length = 64

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.train = Train(Vector(width / 2 - self.length / 2, height / 2 - self.length - 1))

    def step(self) -> None:
        self.train.update(self.canvas.width, self.length)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        match symbol:
            case arcade.key.UP:
                self.train.throttle(0.1)
            case arcade.key.DOWN:
                self.train.throttle(-0.1)
            case _:
                return super().on_key_press(symbol, modifiers)
        return True

    def draw(self, canvas: Canvas) -> None:
        rail = canvas.height / 2
        canvas.line(0, rail, canvas.width, rail)
        for sleeper in range(0, canvas.width - 5, 15):
            canvas.rect(sleeper, rail, 5, 2.5, fill=gray(0))
        x, y = self.train.position
        n = self.length
        canvas.rect(x, y + n * 0.3, n * 0.7, n * 0.45, fill=(200, 60, 60), weight=2)  # boiler
        canvas.rect(x + n * 0.6, y + n * 0.1, n * 0.4, n * 0.65, fill=(60, 60, 160), weight=2)
        canvas.rect(x + n * 0.1, y + n * 0.1, n * 0.12, n * 0.2, fill=gray(40))  # chimney
        for wx in (0.2, 0.5, 0.8):
            canvas.circle(x + n * wx, y + n * 0.87, n * 0.25, fill=gray(60), weight=2)
        train = self.train
        self.status = f"speed {train.velocity.x:.1f}  acceleration {train.acceleration:.1f}"


class MotionRandomAcceleration(MoverSketch):
    title = "Example 1.9: Motion 101 (Velocity and Random Acceleration)"

    def make_mover(self) -> Mover:
        self.rng = random.Random()
        mover = super().make_mover()
        mover.top_speed = 5
        return mover

    def accelerate(self) -> None:
        self.mover.acceleration = random_acceleration(self.rng)


class MotionNoiseAcceleration(MoverSketch):
    title = "Exercise 1.6: Acceleration from Perlin Noise"

    def make_mover(self) -> Mover:
        self.noise_acceleration = NoiseAcceleration(Noise())
        mover = super().make_mover()
        mover.top_speed = 5
        return mover

    def accelerate(self) -> None:
        self.mover.acceleration = self.noise_acceleration()


class AcceleratingTowardMouse(MoverSketch):
    title = "Example 1.10: Accelerating Toward the Mouse"
    help = ("Move the mouse",)

    def make_mover(self) -> Mover:
        mover = super().make_mover()
        mover.top_speed = 5
        return mover

    def accelerate(self) -> None:
        self.mover.acceleration = toward(Vector(*self.mouse), self.mover.position)

    def step(self) -> None:
        self.accelerate()
        self.mover.update()  # no wrapping: the mouse keeps it on the canvas


class AttractionByDistance(AcceleratingTowardMouse):
    title = "Exercise 1.8: Acceleration Stronger with Distance"

    def accelerate(self) -> None:
        reach = max(self.canvas_size)
        self.mover.acceleration = toward_by_distance(
            Vector(*self.mouse), self.mover.position, reach
        )


SKETCHES: tuple[type[Sketch], ...] = (
    BouncingBallNoVectors,
    BouncingBallWithVectors,
    WalkerWithVectors,
    VectorSubtraction,
    VectorMultiplication,
    VectorMagnitude,
    VectorNormalize,
    MotionVelocity,
    MotionConstantAcceleration,
    AccelerateAndBrake,
    MotionRandomAcceleration,
    MotionNoiseAcceleration,
    AcceleratingTowardMouse,
    AttractionByDistance,
)
