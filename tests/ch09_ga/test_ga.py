import random
from collections import Counter

import pytest

from noc.ch09_ga.ecosystem import Flower, World, next_flowers
from noc.ch09_ga.ga import (
    CHARACTERS,
    Population,
    accept_reject,
    coin_crossover,
    exponential_fitness,
    linear_fitness,
    mating_pool,
    midpoint_crossover,
    monkey_attempts,
    mutate,
    random_phrase,
    relay_race,
    two_parents,
)
from noc.ch09_ga.rockets import Box, RocketPopulation, basic_fitness, smart_fitness
from noc.common.vector import Vector


def test_mating_pool_repeats_by_fitness() -> None:
    pool = mating_pool(["a", "b", "c"], [0.5, 0.2, 0.0])
    assert Counter(pool) == {"a": 50, "b": 20}


@pytest.mark.parametrize("select", [relay_race, accept_reject])
def test_weighted_selection_follows_fitness(select: object) -> None:
    rng = random.Random(1)
    items, fitnesses = ["a", "b", "c"], [1.0, 3.0, 0.0]
    picks = Counter(select(items, fitnesses, rng) for _ in range(8000))  # type: ignore[operator]
    assert picks["c"] == 0
    assert picks["b"] / picks["a"] == pytest.approx(3, rel=0.1)


def test_two_unique_parents() -> None:
    rng = random.Random(2)
    for _ in range(200):
        a, b = two_parents(["x", "y", "z"], [10.0, 1.0, 1.0], rng, unique=True)
        assert a != b


def test_crossovers_take_every_gene_from_a_parent() -> None:
    rng = random.Random(3)
    a, b = "aaaaaaaaaa", "bbbbbbbbbb"
    child = "".join(midpoint_crossover(a, b, rng))
    assert child == "a" * child.count("a") + "b" * child.count("b")  # one cut
    mixed = coin_crossover(a, b, rng)
    assert set(mixed) <= {"a", "b"}
    assert len(mixed) == 10


def test_mutation_rate() -> None:
    rng = random.Random(4)
    genes = mutate([0] * 10_000, 0.01, lambda: 1, rng)
    assert sum(genes) == pytest.approx(100, rel=0.25)


def test_fitness_functions() -> None:
    assert linear_fitness("to be", "to be") == 1
    assert linear_fitness("to xx", "to be") == pytest.approx(3 / 5)
    assert exponential_fitness("to be", "to be") == 1
    assert exponential_fitness("xxxxx", "to be") == 0
    # Each extra correct letter doubles (roughly) the exponential fitness.
    assert exponential_fitness("to bx", "to be") > 1.9 * exponential_fitness("to xx", "to be")


def test_random_phrases_use_printable_ascii() -> None:
    phrase = random_phrase(1000, random.Random(5))
    assert set(phrase) <= set(CHARACTERS)
    assert len(CHARACTERS) == 95


@pytest.mark.parametrize("coin", [False, True])
def test_shakespeare_population_solves_the_phrase(coin: bool) -> None:
    population = Population("to be", size=150, coin_flip=coin, rng=random.Random(6))
    for _ in range(500):
        if population.finished:
            break
        population.evolve()
    assert population.finished
    assert population.best == "to be"


def test_dynamic_mutation_drops_as_fitness_rises() -> None:
    population = Population("to be", dynamic_mutation=True, rng=random.Random(7))
    early = population.current_mutation_rate()
    population.phrases = ["to be"] * population.size
    assert population.current_mutation_rate() < early


def test_monkeys_eventually_type_a_short_word() -> None:
    assert monkey_attempts("a", random.Random(8)) < 200


# --- rockets -------------------------------------------------------------------------------
def test_box_and_fitness() -> None:
    target = Box(100, 0, 20, 20)
    assert target.contains(Vector(110, 10))
    assert target.center == Vector(110, 10)
    population = RocketPopulation(Vector(110, 210), target, size=3, rng=random.Random(9))
    rocket = population.rockets[0]
    assert basic_fitness(rocket, target) == pytest.approx(1 / 200**2)
    rocket.record_distance, rocket.finish_counter = 10.0, 100
    base = smart_fitness(rocket, target)
    rocket.hit_obstacle = True
    assert smart_fitness(rocket, target) == pytest.approx(base * 0.1)


def test_rockets_stop_at_obstacles() -> None:
    target = Box(300, 0, 20, 20)
    population = RocketPopulation(
        Vector(320, 220), target, size=20, obstacles=[Box(0, 0, 640, 215)], rng=random.Random(10)
    )
    for _ in range(population.lifespan):
        population.step()
    stuck = [r for r in population.rockets if r.hit_obstacle]
    assert stuck
    assert all(r.body.position.y < 215 for r in stuck)


def test_rockets_improve_over_generations() -> None:
    target = Box(308, 24, 24, 24)
    population = RocketPopulation(Vector(320, 220), target, size=60, rng=random.Random(11))

    def closest() -> float:
        return min(r.record_distance for r in population.rockets)

    for _ in range(population.lifespan):
        population.step()
    first = closest()
    for _ in range(12 * (population.lifespan + 1)):
        population.step()
    for _ in range(population.lifespan):
        population.step()
    assert population.generation >= 12
    assert closest() < first


# --- interactive selection and ecosystem ---------------------------------------------------
def test_flower_phenotype_ranges() -> None:
    flower = Flower([0.0] * 14)
    assert (flower.petal_size, flower.petal_count, flower.stem_length) == (4, 2, 50)
    flower = Flower([1.0] * 14)
    assert flower.petal_color == (255, 255, 255, 255)
    assert flower.center_size == 48


def test_favorite_flower_dominates_the_next_generation() -> None:
    rng = random.Random(12)
    flowers = [Flower([float(i) / 10] * 14) for i in range(8)]
    flowers[3].fitness = 1000
    children = next_flowers(flowers, 0.0, rng)
    assert sum(child.genes == flowers[3].genes for child in children) >= 6


def test_bloops_eat_starve_and_reproduce() -> None:
    world = World(640, 240, size=20, rng=random.Random(13))
    for _ in range(3000):
        world.step()
    assert all(0 <= b.gene <= 1 for b in world.bloops)
    big = world.bloops[0] if world.bloops else None
    if big is not None:
        assert big.radius == pytest.approx(big.gene * 25)
        assert big.max_speed == pytest.approx(15 - big.gene * 15)
