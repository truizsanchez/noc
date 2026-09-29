"""Chapter 11's examples and exercises, in book order."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import arcade
import numpy as np

from noc.ch11_neuroevolution.games import (
    Bird,
    Box,
    Course,
    Ecosystem,
    Flock,
    RocketBrains,
    Seekers,
    sense,
    sensor_offsets,
)
from noc.common.vector import Vector
from noc.common.view import Canvas, Sketch, gray

SPEEDS = {arcade.key.KEY_1: 1, arcade.key.KEY_2: 10, arcade.key.KEY_3: 100}


class SpeedSketch(Sketch):
    """Exercise 11.2's "speeding up time": several simulation steps per drawn frame."""

    help: ClassVar[Sequence[str]] = ("1, 2, 3: run 1, 10 or 100 steps per frame",)

    def __init__(self) -> None:
        super().__init__()
        self.speed = 1

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol in SPEEDS:
            self.speed = SPEEDS[symbol]
            return True
        return super().on_key_press(symbol, modifiers)

    def advance(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        for _ in range(self.speed):
            self.advance()


def draw_pipes(canvas: Canvas, course: Course) -> None:
    for pipe in course.pipes:
        canvas.rect(pipe.x, 0, pipe.w, pipe.top, fill=gray(0), stroke=None)
        canvas.rect(
            pipe.x, pipe.bottom, pipe.w, canvas.height - pipe.bottom, fill=gray(0), stroke=None
        )


class FlappyBird(Sketch):
    title = "Example 11.1: Flappy Bird Clone"
    help = ("Click or Space to flap",)

    def __init__(self) -> None:
        super().__init__()
        self.bird = Bird()
        self.course = Course()
        self.crashes = 0

    def mouse_down(self) -> None:
        self.bird.flap()

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.SPACE:
            self.bird.flap()
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        self.course.update(self.bird.x)
        if any(p.collides(self.bird.x, self.bird.y) for p in self.course.pipes):
            self.crashes += 1
        self.bird.update()

    def draw(self, canvas: Canvas) -> None:
        draw_pipes(canvas, self.course)
        for pipe in self.course.pipes:
            if pipe.collides(self.bird.x, self.bird.y):
                canvas.text("OOPS!", pipe.x, pipe.top + 20, (200, 0, 0), size=14)
        canvas.circle(self.bird.x, self.bird.y, 16, fill=gray(127), weight=2)
        # Exercise 11.1: a score for every pair of pipes passed.
        canvas.text(f"score {self.course.score}", canvas.width - 10, 20, size=14, align="right")


class NeuroFlappy(SpeedSketch):
    title = "Example 11.2: Flappy Bird with Neuroevolution"

    def __init__(self) -> None:
        super().__init__()
        self.flock = Flock()

    def advance(self) -> None:
        self.flock.step()

    def draw(self, canvas: Canvas) -> None:
        flock = self.flock
        draw_pipes(canvas, flock.course)
        for y, alive in zip(flock.y, flock.alive, strict=True):
            if alive:
                canvas.circle(flock.x, float(y), 16, fill=gray(127, 60), stroke=gray(0, 80))
        # Exercise 11.2's overlay.
        canvas.text(f"generation {flock.generation}", 10, 18, size=11)
        canvas.text(f"birds alive {int(flock.alive.sum())}/{flock.size}", 10, 34, size=11)
        canvas.text(f"best lifespan {flock.best_lifespan} frames", 10, 50, size=11)
        canvas.text(f"pipes passed {flock.course.score}", 10, 66, size=11)
        self.status = f"speed x{self.speed}"


def draw_rocket(canvas: Canvas, position: np.ndarray, velocity: np.ndarray, stopped: bool) -> None:
    r = 4.0
    with canvas.pushed():
        canvas.translate(float(position[0]), float(position[1]))
        canvas.rotate(math.atan2(velocity[1], velocity[0]) + math.pi / 2)
        fill = (230, 80, 80, 150) if stopped else gray(200, 150)
        canvas.polygon([(0, -r * 2), (-r, r * 2), (r, r * 2)], fill=fill)


class NeuroRockets(SpeedSketch):
    title = "Example 11.3: Smart Rockets with Neuroevolution"
    help = (
        "Click to move the target   V: also sense velocity (Exercise 11.4)",
        "1, 2, 3: run 1, 10 or 100 steps per frame",
    )

    def __init__(self) -> None:
        super().__init__()
        self.use_velocity = False
        self.restart()

    def restart(self) -> None:
        w, h = self.canvas_size
        self.rockets = RocketBrains(
            Box(w / 2 - 12, 24, 24, 24),
            [Box(w / 2 - 75, h / 2, 150, 10)],
            use_velocity=self.use_velocity,
        )

    def mouse_down(self) -> None:
        x, y = self.mouse
        self.rockets.target = Box(x - 12, y - 12, 24, 24)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.V:
            self.use_velocity = not self.use_velocity
            self.restart()
            return True
        return super().on_key_press(symbol, modifiers)

    def advance(self) -> None:
        self.rockets.step()

    def draw(self, canvas: Canvas) -> None:
        r = self.rockets
        t = r.target
        canvas.rect(t.x, t.y, t.w, t.h, fill=gray(127), weight=2)
        for o in r.obstacles:
            canvas.rect(o.x, o.y, o.w, o.h, fill=gray(175), weight=2)
        for position, velocity, stuck in zip(r.position, r.velocity, r.hit_obstacle, strict=True):
            draw_rocket(canvas, position, velocity, bool(stuck))
        inputs = "position and velocity" if self.use_velocity else "position"
        canvas.text(f"Generation #: {r.generation}", 10, 18, size=11)
        canvas.text(f"Cycles left: {r.lifespan - r.life_counter}", 10, 34, size=11)
        canvas.text(f"Reached the target: {int(r.hit_target.sum())}", 10, 50, size=11)
        self.status = f"inputs: {inputs}   speed x{self.speed}"


class DynamicSteering(SpeedSketch):
    title = "Example 11.4: Dynamic Neuroevolutionary Steering"
    help = (
        "1, 2, 3: run 1, 10 or 100 steps per frame",
        "C: the brain outputs the force's x and y instead of angle and size (restarts)",
    )

    def __init__(self) -> None:
        super().__init__()
        self.seekers = Seekers()

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.C:
            self.seekers = Seekers(components=not self.seekers.components)
            return True
        return super().on_key_press(symbol, modifiers)

    def advance(self) -> None:
        self.seekers.step()

    def draw(self, canvas: Canvas) -> None:
        s = self.seekers
        gx, gy = s.glow.position
        canvas.circle(float(gx), float(gy), s.glow.r * 2, fill=(255, 220, 120), stroke=None)
        r = s.r
        for position, velocity in zip(s.position, s.velocity, strict=True):
            with canvas.pushed():
                canvas.translate(float(position[0]), float(position[1]))
                canvas.rotate(math.atan2(velocity[1], velocity[0]))
                canvas.polygon([(r * 2, 0), (-r * 2, -r), (-r * 2, r)], fill=gray(127), weight=1)
        canvas.text(f"Generation #: {s.generation}", 10, 18, size=11)
        canvas.text(f"Cycles left: {s.lifespan - s.life_counter}", 10, 34, size=11)
        canvas.text(f"Time on the glow: {int(s.fitness.sum())}", 10, 50, size=11)
        encoding = "x/y components" if s.components else "angle and magnitude"
        self.status = f"force as {encoding}   speed x{self.speed}"


class SensingBloop(Sketch):
    title = "Example 11.5: A Bloop with Sensors"
    help = ("Move the bloop with the mouse",)

    def __init__(self) -> None:
        super().__init__()
        self.offsets = sensor_offsets(15, 32)

    def draw(self, canvas: Canvas) -> None:
        food, food_r = Vector(canvas.width / 2, canvas.height / 2), 32.0
        canvas.circle(*food.xy, food_r * 2, fill=(120, 200, 120), stroke=None)
        bloop = Vector(*self.mouse)
        values = sense(bloop, self.offsets, food, food_r)
        for (dx, dy), value in zip(self.offsets, values, strict=True):
            tip = (bloop.x + dx, bloop.y + dy)
            canvas.line(*bloop.xy, *tip, gray(0, 120))
            if value > 0:
                canvas.circle(*tip, 8, fill=(255, 0, 0, int(80 + 175 * value)), stroke=None)
        canvas.circle(*bloop.xy, 32, fill=gray(127), stroke=None)


class NeuroEcosystem(SpeedSketch):
    title = "Example 11.6: A Neuroevolutionary Ecosystem"

    def __init__(self) -> None:
        super().__init__()
        self.ecosystem = Ecosystem()

    def advance(self) -> None:
        self.ecosystem.step()

    def draw(self, canvas: Canvas) -> None:
        e = self.ecosystem
        for (x, y), r in zip(e.food, e.food_r, strict=True):
            canvas.circle(float(x), float(y), float(r) * 2, fill=(120, 200, 120, 150), stroke=None)
        readings = e.readings() if len(e) else np.zeros((0, e.sensors))
        for (x, y), radius, sensed in zip(e.position, e.radius, readings, strict=True):
            for (dx, dy), value in zip(e.offsets, sensed, strict=True):
                canvas.line(float(x), float(y), float(x + dx), float(y + dy), gray(0, 50))
                if value:
                    canvas.circle(float(x + dx), float(y + dy), 4, fill=(255, 0, 0), stroke=None)
            canvas.circle(float(x), float(y), float(radius) * 2, fill=gray(80), stroke=None)
        self.status = f"{len(e)} bloops   speed x{self.speed}"


SKETCHES: tuple[type[Sketch], ...] = (
    FlappyBird,
    NeuroFlappy,
    NeuroRockets,
    DynamicSteering,
    SensingBloop,
    NeuroEcosystem,
)
