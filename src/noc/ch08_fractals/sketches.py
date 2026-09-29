"""Chapter 8's examples and exercises, in book order."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable

import arcade

from noc.ch08_fractals.fractals import (
    Branch,
    CantorLine,
    GrowingTree,
    KochLine,
    branch_weight,
    cantor,
    cantor_generations,
    circles_four_times,
    circles_twice,
    cross_lines,
    koch,
    nested_circles,
    sierpinski,
    snowflake,
    stochastic_tree,
    tree,
    windy_tree,
)
from noc.ch08_fractals.lsystem import PRESETS, LSystem, expand, turtle
from noc.common.mathutils import remap
from noc.common.noise import Noise
from noc.common.vector import Vector
from noc.common.view import Canvas, Sketch, gray


class StillSketch(Sketch):
    """A drawing that doesn't change (the book's ``noLoop()``): drawn once, then kept."""

    background = None

    def __init__(self) -> None:
        super().__init__()
        self.dirty = True

    def draw(self, canvas: Canvas) -> None:
        if self.dirty:
            canvas.background(gray(255))
            self.render(canvas)
            self.dirty = False

    def render(self, canvas: Canvas) -> None:
        raise NotImplementedError


def draw_circles(canvas: Canvas, circles: Iterable[tuple[float, float, float]]) -> None:
    for x, y, r in circles:
        canvas.circle(x, y, r * 2, fill=None, weight=1)


class RecursiveCirclesOnce(StillSketch):
    title = "Example 8.1: Recursive Circles Once"

    def render(self, canvas: Canvas) -> None:
        draw_circles(canvas, nested_circles(canvas.width / 2, canvas.height / 2, canvas.width / 2))


class RecursiveCirclesTwice(StillSketch):
    title = "Example 8.2: Recursive Circles Twice"

    def render(self, canvas: Canvas) -> None:
        draw_circles(canvas, circles_twice(canvas.width / 2, canvas.height / 2, 320))


class RecursiveCirclesFourTimes(StillSketch):
    title = "Example 8.3: Recursive Circles Four Times"

    def render(self, canvas: Canvas) -> None:
        draw_circles(canvas, circles_four_times(canvas.width / 2, canvas.height / 2, 320))


class CrossLines(StillSketch):
    title = "Exercise 8.1: A Recursive Pattern of Lines"

    def render(self, canvas: Canvas) -> None:
        w, h = canvas.width, canvas.height
        for a, b in cross_lines(Vector(w / 4, h / 2), Vector(3 * w / 4, h / 2)):
            canvas.line(*a.xy, *b.xy)


class CantorSet(StillSketch):
    title = "Example 8.4: The Cantor Set"
    canvas_size = (640, 120)

    def render(self, canvas: Canvas) -> None:
        for x, y, length in cantor(10, 10, 620):
            canvas.line(x, y, x + length, y, weight=2)


class CantorObjects(Sketch):
    title = "Exercise 8.4: The Cantor Set with Objects, One Generation per Second"
    canvas_size = (640, 120)

    def __init__(self) -> None:
        super().__init__()
        self.generations = cantor_generations(CantorLine(10, 10, 620), 6)

    def draw(self, canvas: Canvas) -> None:
        shown = min(self.frame_count // 60 + 1, len(self.generations))
        for generation in self.generations[:shown]:
            for line in generation:
                canvas.line(line.x, line.y, line.x + line.length, line.y, weight=2)


def koch_points(lines: list[KochLine]) -> list[tuple[float, float]]:
    """A connected curve as one polyline: the starts of all pieces plus the last end."""
    return [line.start.xy for line in lines] + [lines[-1].end.xy]


class KochCurve(StillSketch):
    title = "Example 8.5: The Koch Curve"

    def render(self, canvas: Canvas) -> None:
        lines = koch([KochLine(Vector(0, 200), Vector(canvas.width, 200))], 5)
        canvas.polyline(koch_points(lines))


class KochSnowflake(StillSketch):
    title = "Exercise 8.2: The Koch Snowflake"

    def render(self, canvas: Canvas) -> None:
        lines = snowflake(Vector(canvas.width / 2, canvas.height / 2 + 15), 120, 4)
        canvas.polygon(koch_points(lines)[:-1], fill=(220, 235, 255))


class AnimatedKoch(Sketch):
    title = "Exercise 8.3: The Koch Curve Drawn from Left to Right"

    def __init__(self) -> None:
        super().__init__()
        self.lines = koch([KochLine(Vector(0, 200), Vector(self.canvas.width, 200))], 5)

    def draw(self, canvas: Canvas) -> None:
        shown = (self.frame_count * 4) % (len(self.lines) + 120)  # a pause when complete
        drawn = self.lines[: max(1, min(shown, len(self.lines)))]
        canvas.polyline(koch_points(drawn), weight=2)
        tip = drawn[-1].end
        canvas.circle(*tip.xy, 8, fill=(220, 60, 60), stroke=None)


class SierpinskiTriangle(StillSketch):
    title = "Exercise 8.5: The Sierpinski Triangle"

    def render(self, canvas: Canvas) -> None:
        h = canvas.height - 20
        side = h * 2 / math.sqrt(3)
        cx = canvas.width / 2
        a, b, c = Vector(cx, 10), Vector(cx - side / 2, 10 + h), Vector(cx + side / 2, 10 + h)
        for triangle in sierpinski(a, b, c, 6):
            canvas.polygon([p.xy for p in triangle], fill=gray(0), stroke=None)


# --- trees ---------------------------------------------------------------------------------
class RecursiveTree(Sketch):
    title = "Example 8.6: A Recursive Tree"
    help = ("Move the mouse left and right to change the angle",)

    def draw(self, canvas: Canvas) -> None:
        angle = remap(self.mouse[0], 0, canvas.width, 0, math.pi / 2)
        canvas.translate(canvas.width / 2, canvas.height)
        self.branch(canvas, 80, angle)

    def branch(self, canvas: Canvas, length: float, angle: float) -> None:
        """The book's version, with the transformation stack: draw, move up, split."""
        canvas.line(0, 0, 0, -length, weight=2)
        canvas.translate(0, -length)
        length *= 0.67
        if length > 2:
            for turn in (angle, -angle):
                with canvas.pushed():
                    canvas.rotate(turn)
                    self.branch(canvas, length, angle)


def draw_branches(
    canvas: Canvas, branches: Iterable[Branch], weight: Callable[[float], float] | None = None
) -> None:
    """Branches as lines, optionally ``weight(length)`` thick."""
    for b in branches:
        canvas.line(*b.start.xy, *b.end.xy, weight=weight(b.length) if weight else 1)


class ThickBranches(Sketch):
    title = "Exercise 8.7: Thick Trunk, Thin Branches"
    help = ("Move the mouse left and right to change the angle",)

    def draw(self, canvas: Canvas) -> None:
        angle = remap(self.mouse[0], 0, canvas.width, 0, math.pi / 2)
        draw_branches(
            canvas, tree(Vector(canvas.width / 2, canvas.height), 80, angle), branch_weight
        )


class StochasticTree(Sketch):
    title = "Example 8.7: A Stochastic Tree"
    help = ("A new tree every second",)

    def draw(self, canvas: Canvas) -> None:
        rng = random.Random(self.frame_count // 60)  # the same tree for a whole second
        draw_branches(canvas, stochastic_tree(Vector(canvas.width / 2, canvas.height), 80, rng))


class GrowingTreeSketch(Sketch):
    title = "Exercise 8.8: A Tree of Branch Objects That Grows"
    help = ("R: plant a new tree",)

    def __init__(self) -> None:
        super().__init__()
        self.tree = GrowingTree.seed(Vector(self.canvas.width / 2, self.canvas.height))

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.tree = GrowingTree.seed(Vector(self.canvas.width / 2, self.canvas.height))
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        self.tree.update()

    def draw(self, canvas: Canvas) -> None:
        for b in self.tree.branches:
            canvas.line(*b.start.xy, *b.end.xy)
        for leaf in self.tree.leaves:
            canvas.circle(*leaf.xy, 4, fill=(50, 120, 50, 150), stroke=None)
        self.status = f"{len(self.tree.branches)} branches, {len(self.tree.leaves)} leaves"


class WindyTree(Sketch):
    title = "Exercise 8.9: A Tree Blowing in Perlin Noise Wind"
    help = ("Click for a new tree",)
    canvas_size = (360, 240)

    def __init__(self) -> None:
        super().__init__()
        self.noise, self.time, self.seed = Noise(), 0.0, 5

    def mouse_down(self) -> None:
        self.time = random.uniform(0, 1000)
        self.seed = random.randrange(1_000_000)

    def step(self) -> None:
        self.time += 0.005

    def draw(self, canvas: Canvas) -> None:
        rng = random.Random(self.seed)  # the same branching every frame; only angles move
        root = Vector(canvas.width / 2, canvas.height)
        branches = windy_tree(root, 60, self.noise, self.time, rng)
        draw_branches(canvas, branches, lambda length: remap(length, 2, 100, 1, 5))


# --- L-systems -----------------------------------------------------------------------------
class LSystemSentences(StillSketch):
    title = "Example 8.8: Simple L-system Sentence Generation"
    canvas_size = (640, 160)

    def render(self, canvas: Canvas) -> None:
        system = LSystem("A", {"A": "AB", "B": "A"})
        for i in range(9):
            canvas.text(f"{i}: {system.sentence}", 4, 20 + i * 16, size=11)
            system.generate()


class LSystemSketch(StillSketch):
    title = "Example 8.9: An L-system"
    help = ("Space: next L-system (Exercise 8.12)   R: regrow",)

    def __init__(self) -> None:
        super().__init__()
        self.index = 0
        self.rng = random.Random()

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.SPACE:
            self.index = (self.index + 1) % len(PRESETS)
        elif symbol != arcade.key.R:
            return super().on_key_press(symbol, modifiers)
        self.dirty = True
        return True

    def render(self, canvas: Canvas) -> None:
        preset = PRESETS[self.index]
        sentence = expand(preset, self.rng)
        segments = list(turtle(sentence, preset.length, math.radians(preset.angle), Vector(0, 0)))
        # Fit the drawing to the canvas: plants grow from the bottom center, curves are centered.
        xs = [p.x for a, b, _ in segments for p in (a, b)]
        ys = [p.y for a, b, _ in segments for p in (a, b)]
        scale = min(
            1.0,
            (canvas.width - 20) / (max(xs) - min(xs) + 1),
            (canvas.height - 20) / (max(ys) - min(ys) + 1),
        )
        offset = Vector(
            canvas.width / 2 - (max(xs) + min(xs)) / 2 * scale,
            canvas.height - 10 - max(ys) * scale,
        )
        for a, b, depth in segments:
            color = (
                (40 + 30 * min(depth, 4), 80 + 30 * min(depth, 4), 40)
                if "plant" in preset.name
                else gray(0)
            )
            canvas.line(*(offset + a * scale).xy, *(offset + b * scale).xy, color)
        self.status = f"{preset.name}: {len(sentence)} symbols, {len(segments)} segments"


SKETCHES: tuple[type[Sketch], ...] = (
    RecursiveCirclesOnce,
    RecursiveCirclesTwice,
    RecursiveCirclesFourTimes,
    CrossLines,
    CantorSet,
    CantorObjects,
    KochCurve,
    KochSnowflake,
    AnimatedKoch,
    SierpinskiTriangle,
    RecursiveTree,
    ThickBranches,
    StochasticTree,
    GrowingTreeSketch,
    WindyTree,
    LSystemSentences,
    LSystemSketch,
)
