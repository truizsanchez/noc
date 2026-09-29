"""Chapter 10's examples and exercises, in book order."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import ClassVar

import arcade
import numpy as np

from noc.ch10_nn.perceptron import (
    EvolvedPerceptrons,
    Perceptron,
    TrainingSet,
    accuracy,
    line,
)
from noc.common.neural import Classifier, NeuralNetwork
from noc.common.vector import Vector
from noc.common.view import Canvas, Sketch, gray


def draw_loss_graph(
    canvas: Canvas, losses: list[float], x: float, y: float, w: float, h: float
) -> None:
    """A little version of TensorFlow's Visor: the loss after every epoch."""
    canvas.rect(x, y, w, h, fill=gray(255, 220), stroke=gray(150))
    canvas.text("loss", x + 4, y + 12, gray(80), size=9)
    if len(losses) > 1:
        top = max(losses)
        points = [
            (x + w * i / (len(losses) - 1), y + h - h * loss / top * 0.9)
            for i, loss in enumerate(losses)
        ]
        canvas.polyline(points, (40, 80, 200), 2)
        canvas.text(f"{losses[-1]:.3f}", x + w - 4, y + 12, gray(80), size=9, align="right")


class PerceptronSketch(Sketch):
    title = "Example 10.1: The Perceptron"
    help: ClassVar[Sequence[str]] = (
        "Gray: guessed above the line   White: below   Red: the perceptron's line (Exercise 10.1)",
        "N: normalize the inputs (Exercise 10.3)   F: train 100 points per frame   R: restart",
    )

    def __init__(self) -> None:
        super().__init__()
        self.normalize, self.fast = False, False
        self.restart()

    def restart(self) -> None:
        rng = random.Random()
        w, h = self.canvas_size
        self.raw = TrainingSet.random(2000, w, h, rng)
        self.answers = [self.raw.answer(p, line) for p in self.raw.points]
        # Normalized inputs are 300 times smaller, so they can learn 300 times faster.
        rate = 0.01 if self.normalize else 0.0001
        self.data = self.raw.normalized(w, h) if self.normalize else self.raw
        self.perceptron = Perceptron.random(3, rate, rng)
        self.count = 0

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        match symbol:
            case arcade.key.N:
                self.normalize = not self.normalize
            case arcade.key.F:
                self.fast = not self.fast
                return True
            case arcade.key.R:
                pass
            case _:
                return super().on_key_press(symbol, modifiers)
        self.restart()
        return True

    def step(self) -> None:
        for _ in range(100 if self.fast else 1):
            i = self.count % len(self.answers)
            self.perceptron.train(self.data.points[i], self.answers[i])
            self.count += 1

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """The book flips y with ``scale(1, -1)`` so that up is positive."""
        return self.canvas.width / 2 + x, self.canvas.height / 2 - y

    def draw(self, canvas: Canvas) -> None:
        w, h = canvas.width, canvas.height
        canvas.line(
            *self.to_screen(-w / 2, line(-w / 2)), *self.to_screen(w / 2, line(w / 2)), weight=2
        )
        weights = np.asarray(self.perceptron.weights)
        guesses = np.asarray(self.data.points) @ weights > 0
        centers = [self.to_screen(x, y) for x, y, _ in self.raw.points]
        canvas.discs(centers, 8, [gray(127) if g else gray(255) for g in guesses])
        # The perceptron's own line, in raw coordinates.
        sx, sy = (w / 2, h / 2) if self.normalize else (1.0, 1.0)
        x0, x1 = -w / 2, w / 2
        y0 = self.perceptron.boundary(x0 / sx) * sy
        y1 = self.perceptron.boundary(x1 / sx) * sy
        canvas.line(*self.to_screen(x0, y0), *self.to_screen(x1, y1), (220, 40, 40), 2)
        score = accuracy(self.perceptron, self.data, self.answers)
        mode = "normalized" if self.normalize else "raw pixels"
        self.status = f"{self.count} points trained ({mode}), accuracy {score:.1%}"


class GeneticPerceptron(PerceptronSketch):
    title = "Exercise 10.2: A Perceptron Trained by a Genetic Algorithm"
    help = ("The best perceptron of each generation; R: restart",)

    def restart(self) -> None:
        super().restart()
        self.data = self.raw.normalized(*self.canvas_size)
        self.normalize = True
        self.evolution = EvolvedPerceptrons(self.data, self.answers)
        self.perceptron = self.evolution.best

    def step(self) -> None:
        if self.frame_count % 10 == 0:
            self.evolution.evolve()
            self.perceptron = self.evolution.best
            self.count = self.evolution.generation

    def draw(self, canvas: Canvas) -> None:
        super().draw(canvas)
        score = accuracy(self.perceptron, self.data, self.answers)
        self.status = f"generation {self.evolution.generation}, best accuracy {score:.1%}"


class XorNetwork(Sketch):
    title = "Putting the “Network” in Neural Network: Learning XOR"
    help = ("Shading: the network's output over the input square   R: new random weights",)

    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        cols, rows = 64, 24
        xs, ys = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
        self.grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
        self.grid_shape = (rows, cols)

    def reset(self) -> None:
        self.network = NeuralNetwork.create((2, 4, 1), output="sigmoid")
        self.losses: list[float] = []

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.reset()
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        for _ in range(20):
            loss = self.network.train_step(self.inputs, self.targets, 2.0)
        if len(self.losses) < 400:
            self.losses.append(loss)

    def draw(self, canvas: Canvas) -> None:
        outputs = self.network.predict(self.grid).reshape(self.grid_shape)
        shade = (outputs * 255).astype(np.uint8)[::-1]  # y up, like a graph
        canvas.pixels(np.repeat(shade[..., None], 3, axis=2))
        w, h = canvas.width, canvas.height
        for (a, b), target in zip(self.inputs, self.targets[:, 0], strict=True):
            x, y = 20 + a * (w - 40), h - 20 - b * (h - 40)
            canvas.circle(x, y, 24, fill=(220, 40, 40) if target else (40, 80, 220), weight=2)
            guess = float(self.network.predict([a, b])[0])
            label_y = y + 30 if b else y - 20  # inside the canvas
            canvas.text(f"{guess:.2f}", x, label_y, (200, 0, 0), size=12, align="center")
        draw_loss_graph(canvas, self.losses, w - 170, 30, 150, 60)


GESTURES = [
    (0.99, 0.02, "right"),
    (0.76, -0.1, "right"),
    (-1.0, 0.12, "left"),
    (-0.9, -0.1, "left"),
    (0.02, 0.98, "down"),
    (-0.2, 0.75, "down"),
    (0.01, -0.9, "up"),
    (-0.1, -0.8, "up"),
]
LABEL_KEYS = {arcade.key.U: "up", arcade.key.D: "down", arcade.key.L: "left", arcade.key.R: "right"}


class GestureClassifier(Sketch):
    title = "Example 10.2: Gesture Classifier"
    help = (
        "Drag a gesture to classify it",
        "Hold U, D, L or R while dragging to add a labeled example and retrain (Exercise 10.4)",
    )
    epochs = 200

    def __init__(self) -> None:
        super().__init__()
        self.data = list(GESTURES)
        self.start: Vector | None = None
        self.label = "training"
        self.retrain()

    def retrain(self) -> None:
        inputs = [(x, y) for x, y, _ in self.data]
        self.classifier = Classifier.create(inputs, [label for *_, label in self.data])
        self.losses: list[float] = []

    def training(self) -> bool:
        return len(self.losses) < self.epochs

    def mouse_down(self) -> None:
        self.start = Vector(*self.mouse)

    def mouse_up(self) -> None:
        if self.start is None:
            return
        direction = (Vector(*self.mouse) - self.start).normalize()
        self.start = None
        if not direction:
            return
        held = [label for key, label in LABEL_KEYS.items() if key in self.held]
        if held:
            self.data.append((direction.x, direction.y, held[0]))
            self.retrain()
        elif not self.training():
            results = self.classifier.classify(direction.xy)
            self.label = f"{results[0][0]} ({results[0][1]:.0%})"

    def step(self) -> None:
        if self.training():
            inputs = [(x, y) for x, y, _ in self.data]
            labels = [label for *_, label in self.data]
            for _ in range(2):  # two epochs per frame: the training takes a few seconds
                if self.training():
                    self.losses.append(self.classifier.train_epoch(inputs, labels, 1.0))
            if not self.training():
                self.label = "ready"

    def draw(self, canvas: Canvas) -> None:
        text = "training" if self.training() else self.label
        canvas.text(
            text, canvas.width / 2, canvas.height / 2, size=48, align="center", baseline="center"
        )
        if self.start is not None:
            canvas.line(*self.start.xy, *self.mouse, weight=8)
        # The dataset as arrows from a corner.
        origin = Vector(60, canvas.height - 60)
        for x, y, _ in self.data:
            tip = origin + Vector(x, y) * 40
            canvas.line(*origin.xy, *tip.xy, gray(0, 120))
            canvas.circle(*tip.xy, 4, fill=gray(0, 120), stroke=None)
        draw_loss_graph(canvas, self.losses, canvas.width - 170, 20, 150, 60)
        self.status = f"{len(self.data)} examples, {len(self.losses)} epochs"


SKETCHES: tuple[type[Sketch], ...] = (
    PerceptronSketch,
    GeneticPerceptron,
    XorNetwork,
    GestureClassifier,
)
