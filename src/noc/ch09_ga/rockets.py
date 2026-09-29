"""Smart rockets (Examples 9.2 and 9.3): the genes are the forces a rocket fires, frame by frame."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from noc.ch09_ga.ga import midpoint_crossover, mutate, relay_race
from noc.common.physics import Mover
from noc.common.vector import Vector

MAX_FORCE = 0.1


def random_force(rng: random.Random) -> Vector:
    return Vector.from_angle(rng.uniform(0, 2 * math.pi), rng.uniform(0, MAX_FORCE))


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle: an obstacle, or the target (Example 9.3)."""

    x: float
    y: float
    w: float
    h: float

    def contains(self, p: Vector) -> bool:
        return self.x < p.x < self.x + self.w and self.y < p.y < self.y + self.h

    @property
    def center(self) -> Vector:
        return Vector(self.x + self.w / 2, self.y + self.h / 2)


@dataclass
class Rocket:
    body: Mover
    genes: list[Vector]
    gene_counter: int = 0
    record_distance: float = math.inf
    finish_counter: int = 0
    hit_obstacle: bool = False
    hit_target: bool = False

    def check_target(self, target: Box) -> None:
        """Example 9.3: remember the closest approach and how long it took to arrive."""
        self.record_distance = min(self.record_distance, self.body.position.dist(target.center))
        if target.contains(self.body.position):
            self.hit_target = True
        if not self.hit_target:
            self.finish_counter += 1

    def run(self, obstacles: list[Box]) -> None:
        if self.hit_obstacle or self.hit_target:
            return  # stopped
        self.body.apply_force(self.genes[self.gene_counter])
        self.gene_counter = (self.gene_counter + 1) % len(self.genes)
        self.body.update()
        self.hit_obstacle = any(o.contains(self.body.position) for o in obstacles)


def basic_fitness(rocket: Rocket, target: Box) -> float:
    """Example 9.2: ``1 / distance²`` at the end of the rocket's life."""
    distance = max(rocket.body.position.dist(target.center), 1.0)
    return 1 / (distance * distance)


def smart_fitness(rocket: Rocket, target: Box) -> float:
    """Example 9.3: reward getting close and getting there fast, to the fourth power;
    obstacles cost 90%, reaching the target doubles it."""
    fitness = (1 / (max(rocket.finish_counter, 1) * max(rocket.record_distance, 1.0))) ** 4
    if rocket.hit_obstacle:
        fitness *= 0.1
    if rocket.hit_target:
        fitness *= 2
    return fitness


@dataclass
class RocketPopulation:
    start: Vector
    target: Box
    size: int = 150
    lifespan: int = 250
    mutation_rate: float = 0.01
    smart: bool = True  # Example 9.3's fitness and obstacles, or Example 9.2's
    obstacles: list[Box] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)
    rockets: list[Rocket] = field(init=False)
    generation: int = 0
    life_counter: int = 0
    record_time: int = field(init=False)

    def __post_init__(self) -> None:
        self.record_time = self.lifespan
        self.rockets = [self.launch([random_force(self.rng) for _ in range(self.lifespan)])
                        for _ in range(self.size)]  # fmt: skip

    def launch(self, genes: list[Vector]) -> Rocket:
        return Rocket(Mover(self.start), genes)

    def step(self) -> None:
        """One frame of the generation's life; the next generation when it's over."""
        if self.life_counter < self.lifespan:
            for rocket in self.rockets:
                if self.smart:
                    rocket.check_target(self.target)
                rocket.run(self.obstacles if self.smart else [])
            if self.smart and any(r.hit_target for r in self.rockets):
                self.record_time = min(self.record_time, self.life_counter)
            self.life_counter += 1
        else:
            self.evolve()

    def fitnesses(self) -> list[float]:
        fitness = smart_fitness if self.smart else basic_fitness
        return [fitness(r, self.target) for r in self.rockets]

    def evolve(self) -> None:
        fitnesses = self.fitnesses()
        genes = [r.genes for r in self.rockets]

        def child() -> list[Vector]:
            a, b = relay_race(genes, fitnesses, self.rng), relay_race(genes, fitnesses, self.rng)
            crossed = midpoint_crossover(a, b, self.rng)
            return mutate(crossed, self.mutation_rate, lambda: random_force(self.rng), self.rng)

        self.rockets = [self.launch(child()) for _ in range(self.size)]
        self.generation += 1
        self.life_counter = 0
