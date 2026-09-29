"""The genetic algorithm's three steps as small functions that work on any list of genes.

Selection picks parents in proportion to their fitness; reproduction combines two parents'
genes (crossover) and changes a few at random (mutation). Genes can be characters, vectors
or numbers: these functions don't care, the caller supplies how to make a random gene.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field


# --- selection -----------------------------------------------------------------------------
def mating_pool[T](items: Sequence[T], fitnesses: Sequence[float], scale: int = 100) -> list[T]:
    """Example 9.1: each item added ``floor(fitness * scale)`` times, to pick from uniformly."""
    return [item for item, f in zip(items, fitnesses, strict=True) for _ in range(int(f * scale))]


def relay_race[T](items: Sequence[T], fitnesses: Sequence[float], rng: random.Random) -> T:
    """The book's ``weightedSelection()``: subtract normalized fitnesses from a random number
    until it runs out; whoever makes it cross zero is picked."""
    total = sum(fitnesses)
    start = rng.random()
    index = 0
    while start > 0 and index < len(items):
        start -= fitnesses[index] / total
        index += 1
    return items[max(index - 1, 0)]


def accept_reject[T](items: Sequence[T], fitnesses: Sequence[float], rng: random.Random) -> T:
    """Exercise 9.2: pick at random, keep with probability fitness / best fitness."""
    best = max(fitnesses)
    while True:
        index = rng.randrange(len(items))
        if rng.random() * best < fitnesses[index]:
            return items[index]


def two_parents[T](
    items: Sequence[T],
    fitnesses: Sequence[float],
    rng: random.Random,
    *,
    unique: bool = False,
) -> tuple[T, T]:
    """Two fitness-weighted picks; ``unique`` (Exercise 9.4) rules out picking one twice."""
    a = relay_race(items, fitnesses, rng)
    if not unique or len(items) < 2:
        return a, relay_race(items, fitnesses, rng)
    others = [(item, f) for item, f in zip(items, fitnesses, strict=True) if item is not a]
    rest, rest_fitness = zip(*others, strict=True)
    if sum(rest_fitness) == 0:
        return a, rng.choice(rest)
    return a, relay_race(rest, rest_fitness, rng)


# --- reproduction --------------------------------------------------------------------------
def midpoint_crossover[T](a: Sequence[T], b: Sequence[T], rng: random.Random) -> list[T]:
    """Genes before a random midpoint from ``a``, the rest from ``b``."""
    midpoint = rng.randrange(len(a))
    return [*a[:midpoint], *b[midpoint:]]


def coin_crossover[T](a: Sequence[T], b: Sequence[T], rng: random.Random) -> list[T]:
    """Exercise 9.5: every gene from either parent with a coin flip."""
    return [x if rng.random() < 0.5 else y for x, y in zip(a, b, strict=True)]


def mutate[T](
    genes: Sequence[T], rate: float, new_gene: Callable[[], T], rng: random.Random
) -> list[T]:
    """Replace each gene with a random one with probability ``rate``."""
    return [new_gene() if rng.random() < rate else gene for gene in genes]


# --- Example 9.1: evolving Shakespeare -----------------------------------------------------
CHARACTERS = "".join(chr(c) for c in range(32, 127))  # the book's random(32, 127)


def random_phrase(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(CHARACTERS) for _ in range(length))


def linear_fitness(phrase: str, target: str) -> float:
    """The fraction of characters in the right place."""
    return sum(a == b for a, b in zip(phrase, target, strict=True)) / len(target)


def exponential_fitness(phrase: str, target: str) -> float:
    """Exercise 9.8: ``2^correct``, normalized to [0, 1] so it fits the mating pool."""
    correct = sum(a == b for a, b in zip(phrase, target, strict=True))
    return float((2**correct - 1) / (2 ** len(target) - 1))


@dataclass
class Population:
    """Exercise 9.6: the Shakespeare GA as a class, with statistics and a stopping point.

    ``dynamic_mutation`` (Exercise 9.7) scales the mutation rate down as the average fitness
    goes up.
    """

    target: str
    size: int = 150
    mutation_rate: float = 0.01
    fitness: Callable[[str, str], float] = linear_fitness
    coin_flip: bool = False
    dynamic_mutation: bool = False
    rng: random.Random = field(default_factory=random.Random)
    phrases: list[str] = field(init=False)
    generation: int = 0

    def __post_init__(self) -> None:
        self.phrases = [random_phrase(len(self.target), self.rng) for _ in range(self.size)]

    def scores(self) -> list[float]:
        return [self.fitness(p, self.target) for p in self.phrases]

    @property
    def best(self) -> str:
        return max(self.phrases, key=lambda p: self.fitness(p, self.target))

    @property
    def average_fitness(self) -> float:
        return sum(self.scores()) / self.size

    @property
    def finished(self) -> bool:
        return self.target in self.phrases

    def current_mutation_rate(self) -> float:
        if not self.dynamic_mutation:
            return self.mutation_rate
        return self.mutation_rate * 2 * (1 - self.average_fitness) + 0.001

    def evolve(self) -> None:
        pool = mating_pool(self.phrases, self.scores())
        if not pool:  # nobody scored at all: pick parents uniformly
            pool = self.phrases
        crossover = coin_crossover if self.coin_flip else midpoint_crossover
        rate = self.current_mutation_rate()

        def new_char() -> str:
            return self.rng.choice(CHARACTERS)

        self.phrases = [
            "".join(
                mutate(
                    crossover(self.rng.choice(pool), self.rng.choice(pool), self.rng),
                    rate,
                    new_char,
                    self.rng,
                )
            )
            for _ in range(self.size)
        ]
        self.generation += 1


def monkey_attempts(target: str, rng: random.Random, limit: int = 10_000_000) -> int:
    """Exercise 9.1: random strings until one equals ``target`` (lowercase letters only)."""
    letters = string.ascii_lowercase
    for attempt in range(1, limit + 1):
        if all(rng.choice(letters) == c for c in target):
            return attempt
    return limit
