"""The perceptron: one neuron, a weighted sum of its inputs and the sign of the result."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from noc.ch09_ga.ga import midpoint_crossover, mutate, relay_race


@dataclass
class Perceptron:
    weights: list[float]
    learning_rate: float = 0.0001

    @classmethod
    def random(cls, inputs: int, learning_rate: float, rng: random.Random) -> Perceptron:
        return cls([rng.uniform(-1, 1) for _ in range(inputs)], learning_rate)

    def feedforward(self, inputs: Sequence[float]) -> int:
        total = sum(x * w for x, w in zip(inputs, self.weights, strict=True))
        return 1 if total > 0 else -1

    def train(self, inputs: Sequence[float], desired: int) -> None:
        """Nudge every weight by ``error * input * learning rate``."""
        error = desired - self.feedforward(inputs)
        self.weights = [
            w + error * x * self.learning_rate for w, x in zip(self.weights, inputs, strict=True)
        ]

    def boundary(self, x: float) -> float:
        """Exercise 10.1: the ``y`` where ``w0 * x + w1 * y + w2 * bias = 0`` (bias input 1)."""
        w0, w1, w2 = self.weights
        return -(w0 * x + w2) / w1 if w1 else 0.0


def line(x: float) -> float:
    """The line the perceptron learns to recognize: ``y = 0.5x + 1``."""
    return 0.5 * x + 1


@dataclass
class TrainingSet:
    """Points in ``[-w/2, w/2] x [-h/2, h/2]``, each with a bias input of 1."""

    points: list[tuple[float, float, float]]

    @classmethod
    def random(cls, count: int, width: float, height: float, rng: random.Random) -> TrainingSet:
        return cls(
            [(rng.uniform(-width / 2, width / 2), rng.uniform(-height / 2, height / 2), 1.0)
             for _ in range(count)]
        )  # fmt: skip

    def answer(self, point: tuple[float, float, float], f: Callable[[float], float]) -> int:
        """Above the line is +1, below is -1."""
        return 1 if point[1] > f(point[0]) else -1

    def normalized(self, width: float, height: float) -> TrainingSet:
        """Exercise 10.3: x and y scaled to [-1, 1]."""
        return TrainingSet([(x / (width / 2), y / (height / 2), b) for x, y, b in self.points])


def accuracy(perceptron: Perceptron, data: TrainingSet, answers: Sequence[int]) -> float:
    """The fraction of points the perceptron classifies right (all at once, with numpy)."""
    guesses = np.where(np.asarray(data.points) @ np.asarray(perceptron.weights) > 0, 1, -1)
    return float(np.mean(guesses == np.asarray(answers)))


@dataclass
class EvolvedPerceptrons:
    """Exercise 10.2: find the weights with a genetic algorithm instead of supervised learning."""

    data: TrainingSet
    answers: list[int]
    size: int = 50
    mutation_rate: float = 0.05
    rng: random.Random = field(default_factory=random.Random)
    population: list[Perceptron] = field(init=False)
    generation: int = 0

    def __post_init__(self) -> None:
        self.population = [Perceptron.random(3, 0, self.rng) for _ in range(self.size)]

    def fitnesses(self) -> list[float]:
        # Squared accuracy, so the best perceptrons reproduce much more often.
        return [accuracy(p, self.data, self.answers) ** 2 for p in self.population]

    @property
    def best(self) -> Perceptron:
        return max(self.population, key=lambda p: accuracy(p, self.data, self.answers))

    def evolve(self) -> None:
        fitnesses = self.fitnesses()
        weights = [p.weights for p in self.population]

        def child() -> Perceptron:
            a, b = (
                relay_race(weights, fitnesses, self.rng),
                relay_race(weights, fitnesses, self.rng),
            )
            genes = mutate(
                midpoint_crossover(a, b, self.rng),
                self.mutation_rate,
                lambda: self.rng.uniform(-1, 1),
                self.rng,
            )
            return Perceptron(genes, 0)

        self.population = [child() for _ in range(self.size)]
        self.generation += 1
