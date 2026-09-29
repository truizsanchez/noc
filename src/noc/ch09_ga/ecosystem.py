"""Interactive selection (Example 9.4) and an evolving ecosystem (Example 9.5)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from noc.ch09_ga.ga import midpoint_crossover, mutate, relay_race
from noc.common.mathutils import remap
from noc.common.noise import Noise
from noc.common.vector import Vector


# --- interactive selection -----------------------------------------------------------------
@dataclass
class Flower:
    """Example 9.4: 14 genes from 0 to 1, read as colors and sizes; the viewer is the fitness."""

    genes: list[float]
    fitness: float = 1.0

    @classmethod
    def random(cls, rng: random.Random) -> Flower:
        return cls([rng.random() for _ in range(14)])

    # The phenotype: what each gene means.
    @property
    def petal_color(self) -> tuple[int, int, int, int]:
        r, g, b, a = self.genes[0:4]
        return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

    @property
    def petal_size(self) -> float:
        return remap(self.genes[4], 0, 1, 4, 24)

    @property
    def petal_count(self) -> int:
        return int(remap(self.genes[5], 0, 1, 2, 16))

    @property
    def center_color(self) -> tuple[int, int, int]:
        r, g, b = self.genes[6:9]
        return (int(r * 255), int(g * 255), int(b * 255))

    @property
    def center_size(self) -> float:
        return remap(self.genes[9], 0, 1, 24, 48)

    @property
    def stem_color(self) -> tuple[int, int, int]:
        r, g, b = self.genes[10:13]
        return (int(r * 255), int(g * 255), int(b * 255))

    @property
    def stem_length(self) -> float:
        return remap(self.genes[13], 0, 1, 50, 100)


def next_flowers(flowers: list[Flower], mutation_rate: float, rng: random.Random) -> list[Flower]:
    """Reproduce in proportion to the fitness the viewer gave by hovering."""
    genes = [f.genes for f in flowers]
    fitnesses = [f.fitness for f in flowers]

    def child() -> Flower:
        a, b = relay_race(genes, fitnesses, rng), relay_race(genes, fitnesses, rng)
        return Flower(mutate(midpoint_crossover(a, b, rng), mutation_rate, rng.random, rng))

    return [child() for _ in flowers]


# --- the evolving ecosystem ----------------------------------------------------------------
@dataclass
class Bloop:
    """Example 9.5: one gene decides both size and speed: big and slow or small and fast."""

    position: Vector
    gene: float
    noise: Noise
    xoff: float
    yoff: float
    health: float = 200.0

    @property
    def max_speed(self) -> float:
        return remap(self.gene, 0, 1, 15, 0)

    @property
    def radius(self) -> float:
        return remap(self.gene, 0, 1, 0, 25)

    def update(self, width: float, height: float) -> None:
        vx = remap(self.noise(self.xoff), 0, 1, -self.max_speed, self.max_speed)
        vy = remap(self.noise(self.yoff), 0, 1, -self.max_speed, self.max_speed)
        self.xoff += 0.01
        self.yoff += 0.01
        x, y = self.position + Vector(vx, vy)
        r = self.radius
        x = width + r if x < -r else -r if x > width + r else x
        y = height + r if y < -r else -r if y > height + r else y
        self.position = Vector(x, y)
        self.health -= 0.2

    @property
    def dead(self) -> bool:
        return self.health < 0


@dataclass
class World:
    width: float
    height: float
    size: int = 20
    rng: random.Random = field(default_factory=random.Random)
    noise: Noise = field(default_factory=Noise)
    bloops: list[Bloop] = field(default_factory=list)
    food: list[Vector] = field(default_factory=list)

    def __post_init__(self) -> None:
        for _ in range(self.size):
            self.born(self.random_position(), self.rng.random())
            self.food.append(self.random_position())

    def random_position(self) -> Vector:
        return Vector(self.rng.uniform(0, self.width), self.rng.uniform(0, self.height))

    def born(self, position: Vector, gene: float) -> None:
        offsets = self.rng.uniform(0, 1000), self.rng.uniform(0, 1000)
        self.bloops.append(Bloop(position, gene, self.noise, *offsets))

    def step(self) -> None:
        if self.rng.random() < 0.001:
            self.food.append(self.random_position())
        for bloop in list(self.bloops):
            bloop.update(self.width, self.height)
            eaten = [f for f in self.food if bloop.position.dist(f) < bloop.radius * 2]
            bloop.health += 100 * len(eaten)
            self.food = [f for f in self.food if f not in eaten]
            if bloop.dead:
                self.bloops.remove(bloop)
                self.food.append(bloop.position)  # a dead bloop becomes food
            elif self.rng.random() < 0.0005:
                # Asexual reproduction: a copy of the gene, occasionally mutated.
                gene = self.rng.random() if self.rng.random() < 0.01 else bloop.gene
                self.born(bloop.position, gene)

    def average_gene(self) -> float:
        return sum(b.gene for b in self.bloops) / len(self.bloops) if self.bloops else 0.0
