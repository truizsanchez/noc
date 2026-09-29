"""The introduction's and chapter 0's examples, exercises and figures, in book order."""

from __future__ import annotations

import random
from typing import ClassVar

import arcade
import numpy as np

from noc.ch00_randomness import walkers
from noc.ch00_randomness.distributions import Histogram, Splatter, accept_reject, normal_pdf
from noc.ch00_randomness.landscapes import (
    Terrain,
    grayscale,
    hue_and_brightness,
    noise_field,
    project,
)
from noc.ch00_randomness.walkers import NoiseWalker, StepRule, Walker
from noc.common.mathutils import clamp, remap
from noc.common.noise import Noise, noise
from noc.common.view import Canvas, Sketch, gray


class ExampleTemplate(Sketch):
    title = "Introduction, Example #.#: Example Title"
    background = None

    def draw(self, canvas: Canvas) -> None:
        x, y = random.uniform(0, canvas.width), random.uniform(0, canvas.height)
        canvas.circle(x, y, 16, fill=gray(0, 25), stroke=gray(0, 50))


# --- random walks -------------------------------------------------------------------------
class WalkerSketch(Sketch):
    """A walker leaving a trail of dots on a canvas that is never cleared."""

    background = None
    bounded: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        bounds = (width, height) if self.bounded else None
        self.walker = Walker(width / 2, height / 2, self.rule(), bounds=bounds)

    def rule(self) -> StepRule:
        return walkers.four_directions

    def step(self) -> None:
        self.walker.step()

    def draw(self, canvas: Canvas) -> None:
        canvas.point(self.walker.x, self.walker.y)


class TraditionalRandomWalk(WalkerSketch):
    title = "Example 0.1: A Traditional Random Walk"


class RandomNumberDistribution(Sketch):
    title = "Example 0.2: A Random-Number Distribution"

    def __init__(self) -> None:
        super().__init__()
        self.histogram = Histogram(20)

    def step(self) -> None:
        self.histogram.add(random.random())

    def draw(self, canvas: Canvas) -> None:
        draw_histogram(canvas, self.histogram)


def draw_histogram(canvas: Canvas, histogram: Histogram) -> None:
    w = canvas.width / histogram.bins
    for i, count in enumerate(histogram.counts):
        canvas.rect(i * w, canvas.height - count, w - 1, count, fill=gray(127), weight=2)


class SkewedWalk(WalkerSketch):
    title = "Exercise 0.1: A Walker That Tends to Move Down and to the Right"

    def rule(self) -> StepRule:
        return walkers.skewed


class WalkerTendsRight(WalkerSketch):
    title = "Example 0.3: A Walker That Tends to Move to the Right"
    bounded = True

    def rule(self) -> StepRule:
        return walkers.tends_right


class WalkTowardMouse(WalkerSketch):
    title = "Exercise 0.3: A 50% Chance of Moving Toward the Mouse"
    help = ("Move the mouse to attract the walker",)

    def rule(self) -> StepRule:
        return walkers.toward(lambda: self.mouse)


# --- the normal distribution ---------------------------------------------------------------
class BellCurves(Sketch):
    title = "Figure 0.2: Two Bell Curves, High and Low Standard Deviation"
    help = ("Left: standard deviation 1.5; right: 0.41 (x from -3 to 3)",)

    def draw(self, canvas: Canvas) -> None:
        half = canvas.width / 2
        for panel, sd in enumerate((1.5, 0.41)):
            points = [
                (panel * half + i, remap(normal_pdf(remap(i, 0, half, -3, 3), 0, sd), 0, 1, 238, 2))
                for i in range(int(half) + 1)
            ]
            canvas.polyline(points, weight=2)
        canvas.line(half, 0, half, canvas.height, gray(200))


class GaussianDistribution(Sketch):
    title = "Example 0.4: A Gaussian Distribution"
    background = None

    def draw(self, canvas: Canvas) -> None:
        # A normal distribution with mean 320 and standard deviation 60.
        x = random.gauss(320, 60)
        canvas.circle(x, 120, 16, fill=gray(0, 10), stroke=None)


class PaintSplatter(Sketch):
    title = "Exercise 0.4: Paint Splatter"
    help = (
        "Up/Down: spread   Left/Right: base hue   +/-: dot size",
        "[/]: hue spread   C: clear",
    )
    background = None

    def __init__(self) -> None:
        super().__init__()
        self.splatter = Splatter()
        self.rng = random.Random()
        self.clear_next = True

    def draw(self, canvas: Canvas) -> None:
        if self.clear_next:
            canvas.background(gray(247))
            self.clear_next = False
        x, y, size, rgb = self.splatter.dot(self.rng, canvas.height)
        with canvas.pushed():
            canvas.translate(canvas.width / 2, canvas.height / 2)
            canvas.scale(canvas.height / 2)
            canvas.circle(x, y, size, fill=(*rgb, 191), stroke=None)
        s = self.splatter
        self.status = (
            f"spread {s.spread:.2f}  size {s.size:.0f}  "
            f"hue {s.hue:.0f}  hue spread {s.hue_spread:.0f}"
        )

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        s = self.splatter
        match symbol:
            case arcade.key.UP | arcade.key.DOWN:
                delta = 0.05 if symbol == arcade.key.UP else -0.05
                s.spread = clamp(s.spread + delta, 0, 0.75)
            case arcade.key.LEFT | arcade.key.RIGHT:
                s.hue = (s.hue + (10 if symbol == arcade.key.RIGHT else -10)) % 360
            case arcade.key.EQUAL | arcade.key.MINUS:
                s.size = clamp(s.size + (2 if symbol == arcade.key.EQUAL else -2), 5, 50)
            case arcade.key.BRACKETLEFT | arcade.key.BRACKETRIGHT:
                delta = 5 if symbol == arcade.key.BRACKETRIGHT else -5
                s.hue_spread = clamp(s.hue_spread + delta, 0, 100)
            case arcade.key.C:
                self.clear_next = True
            case _:
                return super().on_key_press(symbol, modifiers)
        return True


class GaussianWalk(WalkerSketch):
    title = "Exercise 0.5: A Gaussian Random Walk"

    def rule(self) -> StepRule:
        return walkers.gaussian(3)


# --- custom distributions ------------------------------------------------------------------
class AcceptRejectDistribution(Sketch):
    title = "Example 0.5: An Accept-Reject Distribution"

    def __init__(self) -> None:
        super().__init__()
        self.histogram = Histogram(20)
        self.rng = random.Random()

    def step(self) -> None:
        # The probability of picking x is x itself.
        self.histogram.add(accept_reject(lambda x: x, self.rng))

    def draw(self, canvas: Canvas) -> None:
        draw_histogram(canvas, self.histogram)


class QuadraticWalk(WalkerSketch):
    title = "Exercise 0.6: Step Sizes from a Quadratic Accept-Reject Distribution"

    def rule(self) -> StepRule:
        return walkers.quadratic(5)


# --- Perlin noise --------------------------------------------------------------------------
class NoiseVersusRandom(Sketch):
    title = "Figure 0.4: Perlin Noise (Left) and Random Values (Right) over Time"
    canvas_size = (720, 240)

    def __init__(self) -> None:
        super().__init__()
        self.t = 0.0
        self.values = [random.uniform(0, 240) for _ in range(360)]

    def step(self) -> None:
        self.t += 0.01

    def draw(self, canvas: Canvas) -> None:
        smooth = [(i, noise(self.t + i * 0.01) * canvas.height) for i in range(360)]
        n = len(self.values)
        jagged = [(360 + i, self.values[(i + self.frame_count) % n]) for i in range(360)]
        canvas.polyline(smooth, weight=2)
        canvas.polyline(jagged, weight=2)
        canvas.line(360, 0, 360, canvas.height, gray(200))


class PerlinNoiseWalker(Sketch):
    title = "Example 0.6: A Perlin Noise Walker"
    background = None

    def __init__(self) -> None:
        super().__init__()
        self.walker = NoiseWalker(*self.canvas_size)

    def step(self) -> None:
        self.walker.step()

    def draw(self, canvas: Canvas) -> None:
        canvas.circle(self.walker.x, self.walker.y, 48, fill=gray(127), weight=2)


class NoiseStepWalker(WalkerSketch):
    title = "Exercise 0.7: Perlin Noise Mapped to the Step Size"

    def __init__(self) -> None:
        super().__init__()
        self.previous = (self.walker.x, self.walker.y)

    def rule(self) -> StepRule:
        return walkers.noise_steps()

    def draw(self, canvas: Canvas) -> None:
        canvas.line(*self.previous, self.walker.x, self.walker.y)
        self.previous = (self.walker.x, self.walker.y)


class TwoDimensionalNoise(Sketch):
    title = "Figure 0.9: 2D Noise, One Gray Pixel per Noise Value"

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.image = grayscale(noise_field(noise, width, height, (0.01, 0.01)))

    def draw(self, canvas: Canvas) -> None:
        canvas.pixels(self.image)


class NoiseDetail(Sketch):
    title = "Exercises 0.8 and 0.9: Noise Detail, Color and Animation"
    help = (
        "Up/Down: octaves   Left/Right: falloff",
        "[/]: xoff step   -/=: yoff step",
        "A: animate (third noise dimension)   C: color/gray   S: new seed",
    )
    canvas_size = (320, 120)

    def __init__(self) -> None:
        super().__init__()
        self.noise = Noise()
        self.xstep = self.ystep = 0.01
        self.z = 0.0
        self.animate = True
        self.color = True

    def step(self) -> None:
        if self.animate:
            self.z += 0.01

    def draw(self, canvas: Canvas) -> None:
        values = noise_field(
            self.noise, canvas.width, canvas.height, (self.xstep, self.ystep), self.z
        )
        canvas.pixels(hue_and_brightness(values) if self.color else grayscale(values))
        n = self.noise
        self.status = (
            f"octaves {n.octaves}  falloff {n.falloff:.2f}  "
            f"xoff {self.xstep:.2f}  yoff {self.ystep:.2f}"
        )

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        n = self.noise
        match symbol:
            case arcade.key.UP | arcade.key.DOWN:
                n.octaves = int(clamp(n.octaves + (1 if symbol == arcade.key.UP else -1), 1, 10))
            case arcade.key.LEFT | arcade.key.RIGHT:
                delta = 0.05 if symbol == arcade.key.RIGHT else -0.05
                n.falloff = round(clamp(n.falloff + delta, 0, 1), 2)
            case arcade.key.BRACKETLEFT | arcade.key.BRACKETRIGHT:
                delta = 0.01 if symbol == arcade.key.BRACKETRIGHT else -0.01
                self.xstep = round(clamp(self.xstep + delta, 0.01, 0.1), 2)
            case arcade.key.MINUS | arcade.key.EQUAL:
                delta = 0.01 if symbol == arcade.key.EQUAL else -0.01
                self.ystep = round(clamp(self.ystep + delta, 0.01, 0.1), 2)
            case arcade.key.A:
                self.animate = not self.animate
            case arcade.key.C:
                self.color = not self.color
            case arcade.key.S:
                n.seed(random.randrange(1_000_000))
            case _:
                return super().on_key_press(symbol, modifiers)
        return True


class NoiseTerrain(Sketch):
    title = "Exercise 0.10: A Landscape with Elevations from Perlin Noise"

    def __init__(self) -> None:
        super().__init__()
        self.terrain = Terrain(Noise())
        self.theta = 0.0

    def step(self) -> None:
        self.terrain.calculate()
        self.theta += 0.0025

    def draw(self, canvas: Canvas) -> None:
        terrain = self.terrain
        screen = project(terrain.vertices(), self.theta, canvas.width, canvas.height)
        shade = np.clip((terrain.elevations + 120) / 240 * 255, 0, 255)  # map(z, -120, 120, 0, 255)
        quads = []
        for i in range(terrain.cols - 1):
            for j in range(terrain.rows - 1):
                corners = (screen[i, j], screen[i + 1, j], screen[i + 1, j + 1], screen[i, j + 1])
                depth = sum(c[2] for c in corners)
                quads.append((depth, corners, int(shade[i, j])))
        quads.sort(key=lambda quad: -quad[0])  # painter's algorithm: farthest first
        for _, corners, value in quads:
            canvas.polygon([(float(c[0]), float(c[1])) for c in corners], fill=gray(value))


SKETCHES: tuple[type[Sketch], ...] = (
    ExampleTemplate,
    TraditionalRandomWalk,
    RandomNumberDistribution,
    SkewedWalk,
    WalkerTendsRight,
    WalkTowardMouse,
    BellCurves,
    GaussianDistribution,
    PaintSplatter,
    GaussianWalk,
    AcceptRejectDistribution,
    QuadraticWalk,
    NoiseVersusRandom,
    PerlinNoiseWalker,
    NoiseStepWalker,
    TwoDimensionalNoise,
    NoiseDetail,
    NoiseTerrain,
)
