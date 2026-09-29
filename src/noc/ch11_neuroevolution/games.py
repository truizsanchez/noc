"""Neuroevolution: neural networks whose weights evolve with a genetic algorithm.

Every simulation here keeps its agents' state in numpy arrays and their brains in one
:class:`~noc.common.neural.Brains` stack, so a whole population thinks in one call: that's
what makes "speeding up time" (many simulation steps per frame) affordable in Python.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from noc.common.neural import Brains
from noc.common.noise import Noise
from noc.common.vector import Vector

type Array = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

WIDTH, HEIGHT = 640, 240


# --- Flappy Bird ---------------------------------------------------------------------------
@dataclass
class Pipe:
    """Example 11.1: a pair of pipes with a 100-pixel gap, sliding left."""

    top: float
    x: float = WIDTH
    spacing: float = 100.0
    w: float = 20.0
    speed: float = 2.0
    passed: bool = False

    @classmethod
    def random(cls, rng: random.Random) -> Pipe:
        return cls(rng.uniform(0, HEIGHT - 100))

    @property
    def bottom(self) -> float:
        return self.top + self.spacing

    def collides(self, x: float, y: float) -> bool:
        return (y < self.top or y > self.bottom) and self.x < x < self.x + self.w

    def update(self) -> None:
        self.x -= self.speed

    @property
    def offscreen(self) -> bool:
        return self.x < -self.w


@dataclass
class Bird:
    """Example 11.1's bird: gravity, a flap impulse and a little air resistance."""

    y: float = 120.0
    velocity: float = 0.0
    x: float = 50.0
    gravity: float = 0.5
    flap_force: float = -10.0

    def flap(self) -> None:
        self.velocity += self.flap_force

    def update(self) -> None:
        self.velocity += self.gravity
        self.y += self.velocity
        self.velocity *= 0.95  # air resistance, applied after moving as in the book
        if self.y > HEIGHT:
            self.y, self.velocity = HEIGHT, 0.0


@dataclass
class Course:
    """The pipes: a new pair every 100 frames; ``score`` counts pairs passed (Exercise 11.1)."""

    rng: random.Random = field(default_factory=random.Random)
    pipes: list[Pipe] = field(default_factory=list)
    frame: int = 0
    score: int = 0

    def __post_init__(self) -> None:
        if not self.pipes:
            self.pipes.append(Pipe.random(self.rng))

    def update(self, bird_x: float = 50.0) -> None:
        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + pipe.w < bird_x:
                pipe.passed = True
                self.score += 1
        self.pipes = [p for p in self.pipes if not p.offscreen]
        self.frame += 1
        if self.frame % 100 == 0:
            self.pipes.append(Pipe.random(self.rng))

    def next_pipe(self, x: float) -> Pipe:
        """The first pipe the bird hasn't passed yet."""
        return next((p for p in self.pipes if p.x + p.w > x), self.pipes[-1])


@dataclass
class Flock:
    """Example 11.2: birds that flap when their brain says so; fitness is frames survived.

    Inputs: height, velocity, the next gap's top and the distance to it, all divided by the
    canvas size. Outputs: ``flap`` and ``no flap`` (a softmax, like ml5's classification).
    """

    size: int = 200
    mutation_rate: float = 0.01
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    course: Course = field(default_factory=Course)
    brains: Brains = field(init=False)
    y: Array = field(init=False)
    velocity: Array = field(init=False)
    alive: BoolArray = field(init=False)
    fitness: Array = field(init=False)
    generation: int = 0
    x: float = 50.0
    last_best: int = 0  # frames the best bird of the previous generation survived

    def __post_init__(self) -> None:
        self.brains = Brains.random(self.size, (4, 16, 2), self.rng, output="softmax")
        self.reset_birds()

    def reset_birds(self) -> None:
        self.y = np.full(self.size, 120.0)
        self.velocity = np.zeros(self.size)
        self.alive = np.ones(self.size, bool)
        self.fitness = np.zeros(self.size)

    def step(self) -> None:
        self.course.update(self.x)
        pipe = self.course.next_pipe(self.x)
        self.alive &= ~np.array([pipe.collides(self.x, y) for y in self.y])
        inputs = np.stack(
            [
                self.y / HEIGHT,
                self.velocity / HEIGHT,
                np.full(self.size, pipe.top / HEIGHT),
                np.full(self.size, (pipe.x - self.x) / WIDTH),
            ],
            axis=1,
        )
        flap = self.brains.predict(inputs)[:, 0] > 0.5
        flying = self.alive
        self.velocity += np.where(flap & flying, -10.0, 0.0) + np.where(flying, 0.5, 0.0)
        self.y = np.where(flying, self.y + self.velocity, self.y)
        self.velocity = np.where(flying, self.velocity * 0.95, self.velocity)
        self.alive &= (self.y >= 0) & (self.y <= HEIGHT)
        self.fitness += self.alive
        if not self.alive.any():
            self.evolve()

    def evolve(self) -> None:
        self.last_best = self.best_lifespan
        self.brains = self.brains.next_generation(self.fitness, self.rng, self.mutation_rate)
        self.generation += 1
        self.course.pipes = self.course.pipes[-1:]  # the book keeps only the newest pipe
        self.course.score = 0
        self.reset_birds()

    @property
    def best_lifespan(self) -> int:
        return int(self.fitness.max())


# --- steering ------------------------------------------------------------------------------
def steering_force(outputs: Array, max_force: float) -> Array:
    """Two sigmoid outputs as a force: the first times 2π is the angle, the second the size."""
    angle = outputs[:, 0] * 2 * math.pi
    magnitude = outputs[:, 1] * max_force
    return np.stack([np.cos(angle), np.sin(angle)], axis=1) * magnitude[:, None]


def component_force(outputs: Array, max_force: float) -> Array:
    """Two sigmoid outputs as the x and y of the force, each from -max to +max.

    Not the book's encoding: with angle and magnitude, a network whose outputs sit near 0.5
    pushes everyone toward angle pi (left), and turning requires learning a mapping that wraps
    around at 2 pi. Components are a smooth function of the direction to the target; measured
    on Example 11.4, creatures evolve about 20 times higher fitness in 60 generations.
    """
    return (outputs * 2 - 1) * max_force


def limit(vectors: Array, maximum: float) -> Array:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * np.minimum(1.0, maximum / np.maximum(norms, 1e-12))


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    def contains(self, points: Array) -> BoolArray:
        px, py = points[:, 0], points[:, 1]
        return (px > self.x) & (px < self.x + self.w) & (py > self.y) & (py < self.y + self.h)

    @property
    def center(self) -> Array:
        return np.array([self.x + self.w / 2, self.y + self.h / 2])


@dataclass
class RocketBrains:
    """Example 11.3: rockets that decide their thrust from where they are (and, for
    Exercise 11.4, how fast they're going), instead of replaying a fixed list of forces."""

    target: Box
    obstacles: list[Box]
    size: int = 150
    lifespan: int = 300
    mutation_rate: float = 0.01
    use_velocity: bool = False
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    max_speed: float = 4.0
    max_force: float = 1.0
    generation: int = 0
    life_counter: int = 0
    brains: Brains = field(init=False)

    def __post_init__(self) -> None:
        inputs = 4 if self.use_velocity else 2
        self.brains = Brains.random(self.size, (inputs, 16, 2), self.rng)
        self.launch()

    def launch(self) -> None:
        self.position = np.tile([320.0, 220.0], (self.size, 1))
        self.velocity = np.zeros((self.size, 2))
        self.record = np.full(self.size, np.inf)
        self.finish = np.zeros(self.size)
        self.hit_obstacle = np.zeros(self.size, bool)
        self.hit_target = np.zeros(self.size, bool)

    def inputs(self) -> Array:
        columns = [self.position[:, 0] / WIDTH, self.position[:, 1] / HEIGHT]
        if self.use_velocity:
            columns += [self.velocity[:, 0] / self.max_speed, self.velocity[:, 1] / self.max_speed]
        return np.stack(columns, axis=1)

    def step(self) -> None:
        if self.life_counter >= self.lifespan:
            self.evolve()
            return
        distance = np.linalg.norm(self.position - self.target.center, axis=1)
        self.record = np.minimum(self.record, distance)
        self.hit_target |= self.target.contains(self.position)
        self.finish += ~self.hit_target
        moving = ~(self.hit_obstacle | self.hit_target)
        force = steering_force(self.brains.predict(self.inputs()), self.max_force)
        # The solution limits the velocity before adding the force, as in its update().
        velocity = limit(self.velocity, self.max_speed) + force
        self.velocity = np.where(moving[:, None], velocity, self.velocity)
        self.position = np.where(moving[:, None], self.position + self.velocity, self.position)
        for obstacle in self.obstacles:
            self.hit_obstacle |= moving & obstacle.contains(self.position)
        self.life_counter += 1

    def fitness(self) -> Array:
        f = (1 / (np.maximum(self.finish, 1) * np.maximum(self.record, 1))) ** 4
        f = np.where(self.hit_obstacle, f * 0.1, f)
        result: Array = np.where(self.hit_target, f * 2, f)
        return result

    def evolve(self) -> None:
        self.brains = self.brains.next_generation(self.fitness(), self.rng, self.mutation_rate)
        self.generation += 1
        self.life_counter = 0
        self.launch()


@dataclass
class Glow:
    """Example 11.4's moving target: a point wandering on Perlin noise."""

    noise: Noise = field(default_factory=Noise)
    xoff: float = 0.0
    yoff: float = 1000.0
    r: float = 24.0

    @property
    def position(self) -> Array:
        return np.array([self.noise(self.xoff) * WIDTH, self.noise(self.yoff) * HEIGHT])

    def update(self) -> None:
        self.xoff += 0.01
        self.yoff += 0.01


@dataclass
class Seekers:
    """Example 11.4: creatures that learn to chase the glow. Their brain sees the direction
    and distance to it and their own velocity; fitness is the time spent touching it."""

    size: int = 50
    lifespan: int = 250
    mutation_rate: float = 0.1
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    glow: Glow = field(default_factory=Glow)
    max_speed: float = 4.0
    r: float = 4.0
    components: bool = False  # force as x/y instead of angle/magnitude (see component_force)
    generation: int = 0
    life_counter: int = 0
    brains: Brains = field(init=False)

    def __post_init__(self) -> None:
        self.brains = Brains.random(self.size, (5, 16, 2), self.rng)
        self.spawn()

    def spawn(self) -> None:
        self.position = self.rng.random((self.size, 2)) * [WIDTH, HEIGHT]
        self.velocity = np.zeros((self.size, 2))
        self.fitness = np.zeros(self.size)

    def step(self) -> None:
        target = self.glow.position
        offset = target - self.position
        distance = np.linalg.norm(offset, axis=1)
        direction = offset / np.maximum(distance, 1e-12)[:, None]
        inputs = np.column_stack([direction, distance / WIDTH, self.velocity / self.max_speed])
        encode = component_force if self.components else steering_force
        force = encode(self.brains.predict(inputs), 1.0)
        self.velocity = limit(self.velocity + force, self.max_speed)
        self.position += self.velocity
        touching = np.linalg.norm(self.position - target, axis=1) < self.r + self.glow.r
        self.fitness += touching
        self.glow.update()
        self.life_counter += 1
        if self.life_counter > self.lifespan:
            self.brains = self.brains.next_generation(self.fitness, self.rng, self.mutation_rate)
            self.generation += 1
            self.life_counter = 0
            self.spawn()


# --- sensing creatures ---------------------------------------------------------------------
def sensor_offsets(count: int, reach: float) -> Array:
    """Example 11.5: ``count`` whiskers spread evenly around a creature, ``reach`` long."""
    angles = np.arange(count) / count * 2 * math.pi
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * reach


def sense(position: Vector, offsets: Array, food: Vector, food_r: float) -> Array:
    """Example 11.5: each whisker's tip reads 1 at the food's center down to 0 at its edge."""
    tips = np.array(position.xy) + offsets
    d = np.linalg.norm(tips - np.array(food.xy), axis=1)
    return np.where(d < food_r, 1 - d / food_r, 0.0)


@dataclass
class Ecosystem:
    """Example 11.6: bloops with 15 whiskers that sense food, a brain that turns what they
    sense into a steering force, health that drains, and asexual reproduction with mutation.
    There are no generations: whoever finds food lives long enough to reproduce."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    sensors: int = 15
    full_size: float = 12.0
    max_speed: float = 2.0
    food_count: int = 8
    minimum: int = 2  # below this, newcomers with random brains arrive (not in the book)

    def __post_init__(self) -> None:
        self.offsets = sensor_offsets(self.sensors, self.full_size * 1.5)
        self.brains = Brains.random(20, (self.sensors, 16, 2), self.rng)
        self.position = self.rng.random((20, 2)) * [WIDTH, HEIGHT]
        self.velocity = np.zeros((20, 2))
        self.health = np.full(20, 100.0)
        self.food = self.rng.random((self.food_count, 2)) * [WIDTH, HEIGHT]
        self.food_r = np.full(self.food_count, 50.0)

    def __len__(self) -> int:
        return len(self.position)

    def readings(self) -> Array:
        """Each creature's whiskers: 1 if the tip is inside any food, else 0."""
        tips = self.position[:, None, :] + self.offsets[None, :, :]  # (n, sensors, 2)
        d = np.linalg.norm(tips[:, :, None, :] - self.food[None, None, :, :], axis=3)
        return (d < self.food_r[None, None, :]).any(axis=2).astype(np.float64)

    @property
    def radius(self) -> Array:
        return np.clip(self.health / 100 * (self.full_size - 2) + 2, 2, self.full_size)

    def arrive(self, count: int) -> None:
        """Newcomers with random brains, so an extinction doesn't end the simulation."""
        newcomers = Brains.random(count, (self.sensors, 16, 2), self.rng)
        self.brains = newcomers if not len(self) else self.brains.concat(newcomers)
        self.position = np.concatenate(
            [self.position, self.rng.random((count, 2)) * [WIDTH, HEIGHT]]
        )
        self.velocity = np.concatenate([self.velocity, np.zeros((count, 2))])
        self.health = np.concatenate([self.health, np.full(count, 100.0)])

    def step(self) -> None:
        if len(self) < self.minimum:
            self.arrive(self.minimum - len(self))
        force = steering_force(self.brains.predict(self.readings()), 1.0)
        # Eating: every creature touching food gains health and shrinks the food.
        d = np.linalg.norm(self.position[:, None, :] - self.food[None, :, :], axis=2)
        touching = d < self.radius[:, None] + self.food_r[None, :]
        self.health += 0.5 * touching.sum(axis=1)
        self.food_r -= 0.05 * touching.sum(axis=0)
        eaten = self.food_r < 20
        self.food[eaten] = self.rng.random((int(eaten.sum()), 2)) * [WIDTH, HEIGHT]
        self.food_r[eaten] = 50.0
        # Moving, wrapping around the edges, and getting hungry.
        self.velocity = limit(self.velocity + force, self.max_speed)
        self.position = (self.position + self.velocity) % [WIDTH, HEIGHT]
        self.health -= 0.25
        # Dying and reproducing.
        alive = self.health >= 0
        births = alive & (self.rng.random(len(self)) < 0.001)
        keep = np.flatnonzero(alive)
        parents = np.flatnonzero(births)
        brains = self.brains.take(keep)
        if len(parents):
            children = self.brains.take(parents)
            children.mutate(0.1, self.rng)
            brains = brains.concat(children)
        order = np.concatenate([keep, parents])
        self.brains = brains
        self.position = self.position[order]
        self.velocity = np.concatenate([self.velocity[keep], np.zeros((len(parents), 2))])
        self.health = np.concatenate([self.health[keep], np.full(len(parents), 100.0)])
