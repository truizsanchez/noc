"""Chapter 9's examples and exercises, in book order."""

from __future__ import annotations

import math
import random
import string
from collections.abc import Sequence
from typing import ClassVar

import arcade

from noc.ch09_ga.ecosystem import Flower, World, next_flowers
from noc.ch09_ga.ga import Population, exponential_fitness, linear_fitness
from noc.ch09_ga.rockets import Box, Rocket, RocketPopulation
from noc.common.mathutils import remap
from noc.common.vector import Vector
from noc.common.view import Canvas, Sketch, gray

TARGET_PHRASE = "to be or not to be"


class MonkeysTyping(Sketch):
    title = "Exercise 9.1: How Long Until Random Typing Produces “cat”?"
    help = ("200 random three-letter words per frame; R: start over",)

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random()
        self.attempts, self.found = 0, 0
        self.recent: list[str] = []

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.attempts, self.found = 0, 0
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        if self.found:
            return
        for _ in range(200):
            word = "".join(self.rng.choice(string.ascii_lowercase) for _ in range(3))
            self.attempts += 1
            self.recent = [*self.recent[-39:], word]
            if word == "cat":
                self.found = self.attempts
                return

    def draw(self, canvas: Canvas) -> None:
        for row in range(4):
            words = " ".join(self.recent[row * 10 : row * 10 + 10])
            canvas.text(words, 12, 40 + row * 24, size=14)
        message = (
            f"“cat” after {self.found:,} tries" if self.found else f"{self.attempts:,} tries so far"
        )
        canvas.text(message, 12, 180, size=16)
        canvas.text(f"expected on average: 26³ = {26**3:,}", 12, 210, size=12)


class Shakespeare(Sketch):
    title = "Example 9.1: Genetic Algorithm for Evolving Shakespeare"

    def __init__(self) -> None:
        super().__init__()
        self.population = Population(TARGET_PHRASE)

    def step(self) -> None:
        if not self.population.finished:
            self.population.evolve()

    def draw(self, canvas: Canvas) -> None:
        phrases = self.population.phrases
        per_line = 4
        for row in range(len(phrases) // per_line)[:16]:
            chunk = phrases[row * per_line : row * per_line + per_line]
            canvas.text("    ".join(chunk), 12, 14 + row * 14, size=9)


class ShakespeareReport(Sketch):
    title = "Exercise 9.6: The Shakespeare GA, with a Report"
    help = (
        "C: coin-flip crossover (9.5)   E: exponential fitness (9.8)",
        "M: dynamic mutation rate (9.7)   R: start over",
    )

    def __init__(self) -> None:
        super().__init__()
        self.coin, self.exponential, self.dynamic = False, False, False
        self.restart()

    def restart(self) -> None:
        self.population = Population(
            TARGET_PHRASE,
            fitness=exponential_fitness if self.exponential else linear_fitness,
            coin_flip=self.coin,
            dynamic_mutation=self.dynamic,
        )

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        match symbol:
            case arcade.key.C:
                self.coin = not self.coin
            case arcade.key.E:
                self.exponential = not self.exponential
            case arcade.key.M:
                self.dynamic = not self.dynamic
            case arcade.key.R:
                pass
            case _:
                return super().on_key_press(symbol, modifiers)
        self.restart()
        return True

    def step(self) -> None:
        if not self.population.finished:
            self.population.evolve()

    def draw(self, canvas: Canvas) -> None:
        p = self.population
        canvas.text("Best phrase:", 12, 30, size=12)
        canvas.text(p.best, 12, 70, size=28)
        lines = [
            f"total generations: {p.generation}",
            f"average fitness: {p.average_fitness:.2f}",
            f"population: {p.size}",
            f"mutation rate: {p.current_mutation_rate() * 100:.1f}%",
            "crossover: " + ("coin flip" if self.coin else "midpoint"),
            "fitness: " + ("exponential" if self.exponential else "linear"),
        ]
        for i, line in enumerate(lines):
            canvas.text(line, 12, 110 + i * 18, size=11)
        if p.finished:
            canvas.text("solved!", 400, 200, (0, 140, 0), size=20)
        for i, phrase in enumerate(p.phrases[:12]):
            canvas.text(phrase, 400, 20 + i * 14, gray(120), size=10)


# --- smart rockets -------------------------------------------------------------------------
def draw_rocket(canvas: Canvas, rocket: Rocket) -> None:
    r = 4.0
    with canvas.pushed():
        canvas.translate(*rocket.body.position.xy)
        canvas.rotate(rocket.body.velocity.heading() + math.pi / 2)
        canvas.rect(-r / 2, r * 2, r / 2, r, fill=gray(0), stroke=None, center=True)
        canvas.rect(r / 2, r * 2, r / 2, r, fill=gray(0), stroke=None, center=True)
        fill = (230, 80, 80, 150) if rocket.hit_obstacle else gray(200, 150)
        canvas.polygon([(0, -r * 2), (-r, r * 2), (r, r * 2)], fill=fill)


class RocketSketch(Sketch):
    help: ClassVar[Sequence[str]] = (
        "Click to move the target   F: fast-forward (a generation per frame)",
    )
    smart = False
    population_size = 50

    def __init__(self) -> None:
        super().__init__()
        self.fast = False
        self.population = self.make_population()

    def make_population(self) -> RocketPopulation:
        w = self.canvas.width
        return RocketPopulation(
            Vector(w / 2, 220),
            Box(w / 2 - 12, 24, 24, 24),
            size=self.population_size,
            smart=self.smart,
        )

    def mouse_down(self) -> None:
        x, y = self.mouse
        self.population.target = Box(x - 12, y - 12, 24, 24)
        self.population.record_time = self.population.lifespan

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.F:
            self.fast = not self.fast
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        for _ in range(self.population.lifespan + 1 if self.fast else 1):
            self.population.step()

    def draw(self, canvas: Canvas) -> None:
        p = self.population
        t = p.target
        if self.smart:
            canvas.rect(t.x, t.y, t.w, t.h, fill=gray(127), weight=2)
        else:
            canvas.circle(*t.center.xy, 24, fill=gray(127), weight=2)
        for o in p.obstacles:
            canvas.rect(o.x, o.y, o.w, o.h, fill=gray(175), weight=2)
        for rocket in p.rockets:
            draw_rocket(canvas, rocket)
        canvas.text(f"Generation #: {p.generation}", 10, 18, size=11)
        canvas.text(f"Cycles left: {p.lifespan - p.life_counter}", 10, 36, size=11)
        if self.smart:
            canvas.text(f"Record cycles: {p.record_time}", 10, 54, size=11)


class SmartRockets(RocketSketch):
    title = "Example 9.2: Smart Rockets"


class SmarterRockets(RocketSketch):
    title = "Example 9.3: Smarter Rockets"
    help = (
        "Click to move the target   F: fast-forward (a generation per frame)",
        "O: a harder obstacle course (Exercise 9.9)",
    )
    smart = True
    population_size = 150

    def make_population(self) -> RocketPopulation:
        population = super().make_population()
        w, h = self.canvas_size
        population.obstacles = [Box(w / 2 - 75, h / 2, 150, 10)]
        return population

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.O:
            w = self.canvas.width
            self.population = self.make_population()
            self.population.obstacles = [
                Box(w / 2 - 160, 150, 200, 10),
                Box(w / 2 - 40, 90, 200, 10),
                Box(w / 2 - 10, 40, 10, 50),
            ]
            return True
        return super().on_key_press(symbol, modifiers)


# --- interactive selection and ecosystem ---------------------------------------------------
class InteractiveSelection(Sketch):
    title = "Example 9.4: Interactive Selection"
    help = ("Hover over the flowers you like (each frame adds fitness)", "Space: evolve")

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random()
        self.flowers = [Flower.random(self.rng) for _ in range(8)]
        self.generation = 0

    def slot(self, i: int) -> tuple[float, float]:
        return 40 + i * 80, 120

    def hovered(self) -> int | None:
        x, y = self.mouse
        for i in range(len(self.flowers)):
            cx, cy = self.slot(i)
            if abs(x - cx) < 35 and abs(y - cy) < 70:
                return i
        return None

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.SPACE:
            self.flowers = next_flowers(self.flowers, 0.05, self.rng)
            self.generation += 1
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        if (i := self.hovered()) is not None:
            self.flowers[i].fitness += 0.25

    def draw(self, canvas: Canvas) -> None:
        hovered = self.hovered()
        for i, flower in enumerate(self.flowers):
            x, y = self.slot(i)
            fill = gray(0, 64) if i == hovered else None
            canvas.rect(x, y, 70, 140, fill=fill, weight=1, center=True)
            with canvas.pushed():
                canvas.translate(x, y + 70 - flower.stem_length)
                canvas.line(0, 0, 0, flower.stem_length, flower.stem_color, 4)
                for k in range(flower.petal_count):
                    angle = remap(k, 0, flower.petal_count, 0, 2 * math.pi)
                    size = flower.petal_size
                    petal = Vector.from_angle(angle, size)
                    canvas.circle(*petal.xy, size, fill=flower.petal_color, stroke=None)
                canvas.circle(0, 0, flower.center_size, fill=flower.center_color, stroke=None)
            canvas.text(f"{int(flower.fitness)}", x, y + 90, size=10, align="center")
        canvas.text(f"Generation {self.generation}", 12, canvas.height - 10, size=11)


class EvolvingEcosystem(Sketch):
    title = "Example 9.5: An Evolving Ecosystem"
    help = ("Click or drag to add bloops   F: fast-forward x20",)

    def __init__(self) -> None:
        super().__init__()
        self.world = World(*self.canvas_size)
        self.fast = False

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.F:
            self.fast = not self.fast
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        if self.mouse_pressed:
            self.world.born(Vector(*self.mouse), self.world.rng.random())
        for _ in range(20 if self.fast else 1):
            self.world.step()

    def draw(self, canvas: Canvas) -> None:
        for f in self.world.food:
            canvas.rect(*f.xy, 8, 8, fill=gray(175), center=True)
        for bloop in self.world.bloops:
            alpha = int(min(max(bloop.health, 0), 255))
            canvas.circle(
                *bloop.position.xy, bloop.radius * 2, fill=gray(0, alpha), stroke=gray(0, alpha)
            )
        w = self.world
        self.status = (
            f"{len(w.bloops)} bloops, {len(w.food)} food, average size gene {w.average_gene():.2f}"
        )


SKETCHES: tuple[type[Sketch], ...] = (
    MonkeysTyping,
    Shakespeare,
    ShakespeareReport,
    SmartRockets,
    SmarterRockets,
    InteractiveSelection,
    EvolvingEcosystem,
)
