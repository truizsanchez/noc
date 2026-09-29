"""A small fully connected neural network in numpy, standing in for ml5.js.

``NeuralNetwork((2, 16, 4))`` has 2 inputs, a hidden layer of 16 neurons and 4 outputs. Each
layer computes ``activation(inputs @ weights + bias)``; hidden layers use the sigmoid (or
ReLU), and the output layer depends on the task:

- ``"softmax"`` for classification: outputs are confidences that add up to 1.
- ``"sigmoid"`` for yes/no outputs (XOR, Flappy Bird's flap).
- ``"linear"`` for regression.

Training is gradient descent with backpropagation. Paired with its natural loss (cross-entropy
for softmax and sigmoid, squared error for linear), every output type has the same gradient at
the output layer, ``prediction - target``, which is why one ``train_step`` serves all three.

For neuroevolution (chapter 11) the network can also be copied, mutated and crossed over,
like ml5.js's ``neuroEvolution`` networks.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]
type Output = Literal["softmax", "sigmoid", "linear"]
type Hidden = Literal["sigmoid", "relu"]


def sigmoid(x: Array) -> Array:
    return np.asarray(1 / (1 + np.exp(-np.clip(x, -500, 500))))


def softmax(x: Array) -> Array:
    shifted = np.exp(x - x.max(axis=-1, keepdims=True))
    return np.asarray(shifted / shifted.sum(axis=-1, keepdims=True))


@dataclass
class NeuralNetwork:
    sizes: tuple[int, ...]
    weights: list[Array]
    biases: list[Array]
    output: Output = "softmax"
    hidden: Hidden = "sigmoid"

    @classmethod
    def create(
        cls,
        sizes: Sequence[int],
        rng: np.random.Generator | None = None,
        output: Output = "softmax",
        hidden: Hidden = "sigmoid",
    ) -> NeuralNetwork:
        """Random weights scaled by the layer size (Xavier initialization), zero biases."""
        rng = rng or np.random.default_rng()
        weights = [
            rng.normal(0, np.sqrt(1 / n_in), (n_in, n_out))
            for n_in, n_out in itertools.pairwise(sizes)
        ]
        biases = [np.zeros(n) for n in sizes[1:]]
        return cls(tuple(sizes), weights, biases, output, hidden)

    # --- using the network ------------------------------------------------------------------
    def _hidden(self, x: Array) -> Array:
        return sigmoid(x) if self.hidden == "sigmoid" else np.maximum(x, 0)

    def _output(self, x: Array) -> Array:
        match self.output:
            case "softmax":
                return softmax(x)
            case "sigmoid":
                return sigmoid(x)
            case "linear":
                return x

    def layers(self, inputs: Array) -> list[Array]:
        """The activations of every layer, inputs first (what backpropagation needs)."""
        activations = [inputs]
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = activations[-1] @ w + b
            activations.append(self._output(z) if i == last else self._hidden(z))
        return activations

    def predict(self, inputs: Sequence[float] | Array) -> Array:
        """Outputs for one input vector, or for a batch (one row per input)."""
        return self.layers(np.asarray(inputs, np.float64))[-1]

    # --- supervised learning ----------------------------------------------------------------
    def loss(self, inputs: Array, targets: Array) -> float:
        """Summed over the outputs, averaged over the samples."""
        p = self.predict(inputs)
        match self.output:
            case "softmax":
                return float(-np.mean(np.sum(targets * np.log(p + 1e-12), axis=1)))
            case "sigmoid":
                both = targets * np.log(p + 1e-12) + (1 - targets) * np.log(1 - p + 1e-12)
                return float(-np.mean(np.sum(both, axis=1)))
            case "linear":
                return float(np.mean(np.sum((p - targets) ** 2, axis=1)) / 2)

    def train_step(self, inputs: Array, targets: Array, learning_rate: float) -> float:
        """One step of gradient descent on the whole batch; returns the loss before it."""
        activations = self.layers(inputs)
        n = len(inputs)
        delta = (activations[-1] - targets) / n  # the shared output gradient
        loss = self.loss(inputs, targets)
        for i in reversed(range(len(self.weights))):
            grad_w = activations[i].T @ delta
            grad_b = delta.sum(axis=0)
            if i > 0:  # pass the error back through this layer and the hidden activation
                a = activations[i]
                slope = a * (1 - a) if self.hidden == "sigmoid" else (a > 0).astype(np.float64)
                delta = (delta @ self.weights[i].T) * slope
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
        return loss

    # --- neuroevolution ---------------------------------------------------------------------
    def copy(self) -> NeuralNetwork:
        return NeuralNetwork(
            self.sizes,
            [w.copy() for w in self.weights],
            [b.copy() for b in self.biases],
            self.output,
            self.hidden,
        )

    def mutate(self, rate: float, rng: np.random.Generator, amount: float = 0.1) -> None:
        """Nudge each weight and bias, with probability ``rate``, by Gaussian noise."""
        for array in (*self.weights, *self.biases):
            mask = rng.random(array.shape) < rate
            np.add(array, mask * rng.normal(0, amount, array.shape), out=array)

    def crossover(self, other: NeuralNetwork, rng: np.random.Generator) -> NeuralNetwork:
        """A child taking each weight from one parent or the other with a coin flip."""
        child = self.copy()
        for mine, theirs in zip(
            (*child.weights, *child.biases), (*other.weights, *other.biases), strict=True
        ):
            pick = rng.random(mine.shape) < 0.5
            mine[pick] = theirs[pick]
        return child

    # --- saving -----------------------------------------------------------------------------
    def to_json(self) -> str:
        """Exercise 10.5's ``save()``: the model as JSON."""
        return json.dumps(
            {
                "sizes": self.sizes,
                "output": self.output,
                "hidden": self.hidden,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> NeuralNetwork:
        data = json.loads(text)
        return cls(
            tuple(data["sizes"]),
            [np.array(w) for w in data["weights"]],
            [np.array(b) for b in data["biases"]],
            data["output"],
            data["hidden"],
        )


@dataclass
class Normalizer:
    """ml5's ``normalizeData()``: scale each input column to [0, 1] by its training range."""

    low: Array = field(default_factory=lambda: np.zeros(0))
    high: Array = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def fit(cls, data: Array) -> Normalizer:
        return cls(data.min(axis=0), data.max(axis=0))

    def __call__(self, data: Sequence[float] | Array) -> Array:
        span = np.where(self.high > self.low, self.high - self.low, 1.0)
        return np.asarray((np.asarray(data, np.float64) - self.low) / span)


@dataclass
class Classifier:
    """ml5's ``neuralNetwork({task: "classification"})``: labels in, confidences out."""

    labels: list[str]
    network: NeuralNetwork
    normalizer: Normalizer

    @classmethod
    def create(
        cls,
        inputs: Sequence[Sequence[float]],
        labels: Sequence[str],
        hidden: int = 16,
        rng: np.random.Generator | None = None,
    ) -> Classifier:
        names = sorted(set(labels))
        network = NeuralNetwork.create((len(inputs[0]), hidden, len(names)), rng)
        return cls(names, network, Normalizer.fit(np.asarray(inputs, np.float64)))

    def one_hot(self, labels: Sequence[str]) -> Array:
        targets = np.zeros((len(labels), len(self.labels)))
        for row, label in enumerate(labels):
            targets[row, self.labels.index(label)] = 1
        return targets

    def train_epoch(
        self, inputs: Sequence[Sequence[float]], labels: Sequence[str], learning_rate: float = 0.5
    ) -> float:
        """One pass over the whole dataset (an epoch); returns the loss."""
        x = self.normalizer(np.asarray(inputs, np.float64))
        return self.network.train_step(x, self.one_hot(labels), learning_rate)

    def classify(self, inputs: Sequence[float]) -> list[tuple[str, float]]:
        """``(label, confidence)`` pairs, most confident first, like ml5's results."""
        confidences = self.network.predict(self.normalizer(inputs))
        return sorted(zip(self.labels, confidences.tolist(), strict=True), key=lambda r: -r[1])
