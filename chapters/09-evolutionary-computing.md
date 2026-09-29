# Chapter 9 — Evolutionary Computing

Run: `uv run python -m noc.ch09_ga` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Exercise 9.1 | how long random typing takes to produce "cat" |
| 2 | Example 9.1 | the GA evolving "to be or not to be", all 150 phrases |
| 3 | Exercise 9.6 | the same GA with a report and a stop; coin-flip crossover (9.5), exponential fitness (9.8), dynamic mutation (9.7) |
| 4 | Example 9.2 | smart rockets evolving toward a target (click to move it) |
| 5 | Example 9.3 | smarter rockets: obstacles, record distance, time to target; `O` for a harder course (9.9) |
| 6 | Example 9.4 | interactive selection: hovering is the fitness |
| 7 | Example 9.5 | an evolving ecosystem of bloops |

`F` fast-forwards the rockets (a whole generation per frame) and the ecosystem (20 steps per
frame).

## Book sections → code

| Book section | Code |
|---|---|
| Step 1: Population Creation | `ch09_ga.ga.random_phrase`, `Population` |
| Step 2: Selection (mating pool, relay race); Exercises 9.2, 9.4 | `mating_pool`, `relay_race`, `accept_reject`, `two_parents(unique=True)` |
| Step 3: Reproduction; Exercise 9.5 | `midpoint_crossover`, `coin_crossover`, `mutate` |
| Customizing GAs: fitness functions, genotype and phenotype | `linear_fitness`, `exponential_fitness`, `Flower`'s properties |
| Evolving Forces: Smart Rockets; Making Improvements | `ch09_ga.rockets` (`Rocket`, `RocketPopulation`, `basic_fitness`, `smart_fitness`, `Box`) |
| Interactive Selection | `ch09_ga.ecosystem.Flower`, `next_flowers` |
| Ecosystem Simulation | `Bloop`, `World` |

## Design decisions and deviations

- **Generic GA operators.** Selection, crossover and mutation are functions over any sequence
  of genes (PEP 695 generics: `def mutate[T](genes: Sequence[T], ...)`), so the same code
  evolves strings, force vectors and flower genes. The book writes them as `DNA` methods for
  each example.
- **Genes are plain data**: a phrase is a `str`, a rocket's genes a `list[Vector]`, a flower's
  a `list[float]`, a bloop's a single `float`. The phenotype is computed from the genes where
  it's used (`Flower.petal_size`, `Bloop.radius`), which is the book's genotype/phenotype
  distinction made explicit.
- **Exercise 9.1** counts attempts at "cat" with lowercase letters only, so the expected
  number of tries (26³ = 17,576) is shown next to the count; a word is abandoned at its first
  wrong letter, which doesn't change the odds.
- **Exercise 9.8** normalizes `2^correct` to [0, 1] so it fits the mating pool; **Exercise
  9.7** scales the mutation rate by `2 × (1 − average fitness)`.
- **Rocket fitness to the fourth power** makes numbers like 10⁻²⁰; they're only ever compared
  relative to each other (normalized in selection), so floats handle them fine.
- **Fast-forward** runs a whole rocket generation (or 20 ecosystem steps) per frame, since the
  interesting part of a GA is often tens of generations away.
- Not ported: Exercises 9.3 (a pen-and-paper question), 9.10–9.14 (open-ended projects).

## Tests

`tests/ch09_ga` checks the mating pool, relay-race and accept-reject selection statistics,
unique parents, both crossovers, the mutation rate, the fitness functions, that the Shakespeare
GA solves a short phrase with either crossover, dynamic mutation, the rockets' fitness and
obstacles and that they get closer over generations, the flower phenotype and selection, and
the ecosystem.
