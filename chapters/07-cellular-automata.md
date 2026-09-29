# Chapter 7 — Cellular Automata

Run: `uv run python -m noc.ch07_ca` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Example 7.1 | Wolfram's elementary CA, rule 90; a random rule once the canvas is full (Exercise 7.1), `S` for a random first row (Exercise 7.2) |
| 2 | Exercise 7.4 | an elementary CA scrolling forever (rule 30; `R`, `Up`/`Down` change it) |
| 3 | Exercise 7.10 | an elementary CA with float states |
| 4 | Example 7.2 | the Game of Life: wraparound (Exercise 7.6), two swapped buffers (Exercise 7.7), editing and known patterns (Exercise 7.5) |
| 5 | Example 7.3 | object-oriented cells, colored by what just changed |
| 6 | Exercise 7.8 | a hexagonal CA (rule B2/S34) |
| 7 | Exercise 7.9 | a probabilistic Game of Life |
| 8 | Exercise 7.11 | one cell per pixel: a cyclic CA |
| 9 | Exercise 7.12 | cells colored by how long they've been alive or dead |

## Book sections → code

| Book section | Code |
|---|---|
| Elementary Cellular Automata; Defining Rulesets | `ch07_ca.automata.ruleset`, `elementary_step`, `single_seed` |
| Drawing an Elementary CA | `ElementaryCA`, `History` (scrolling) |
| The Game of Life: rules and implementation | `neighbor_counts`, `life_step`, `DoubleBuffer` |
| Object-Oriented Cells | `Cell`, `CellBoard` |
| Variations: nonrectangular, probabilistic, continuous, image processing, historical | `hex_life_step`, `probabilistic_step`, `continuous_step`, `cyclic_step`, `Ages` |

## Design decisions and deviations

- **Whole generations with numpy.** Grids are arrays indexed `[row, column]` like images, and
  a generation is a handful of array operations: `elementary_step` builds each cell's
  neighborhood number from shifted copies and looks it up in the rule; `neighbor_counts` adds
  the eight shifted boards. `Cell`/`CellBoard` keep the book's object-oriented version (and a
  test checks both agree generation by generation).
- **Rules as numbers.** `ruleset(90)` returns the book's `[0, 1, 0, 1, 1, 0, 1, 0]`, but the
  step works from the rule number's bits directly (`(rule >> index) & 1`), which is the same
  table read from the other end.
- **Edges.** As in the book, the edge cells of an elementary CA never change and the Game of
  Life's border stays dead; `wrap=True` gives Exercise 7.6's torus.
- **Grids are drawn as images** (`Canvas.pixels(..., smooth=False)` with one block per cell,
  plus grid lines), not one square per cell: 2,400 stroked squares per frame would dominate
  the frame time. Example 7.1 still draws squares, one new row per frame, on a canvas that
  keeps the old ones.
- **Exercise 7.8's solution only draws random hexagons.** Ours evolves: an "odd-r" offset
  grid (odd rows shifted half a cell) with six neighbors each and the Life-like hexagonal
  rule B2/S34.
- **Exercise 7.9** uses the book's two probabilistic death rules and adds a 70% birth chance
  of our own (the exercise invites inventing rules).
- **Exercise 7.10**: a cell counts as on above 0.5, the elementary rule gives its target, and
  the state moves 40% of the way there, so gradients appear between the stripes.
- **Exercise 7.11** is Griffeath's cyclic cellular automaton, a classic "pixel is a cell"
  system: a pixel advances to the next of 14 colors when a neighbor already has it.
- Not ported: Exercises 7.3, 7.13 and 7.14 (open-ended), and the Wolfram classes are
  explored through the rule keys in Exercise 7.4's sketch.

## Tests

`tests/ch07_ca` checks rulesets and specific rules (90's Sierpinski rows, every neighborhood
of rule 30), fixed and wrapping edges, the history, float states, the blinker and a glider
that crosses a wrapping board back to its start, the double buffer against fresh arrays, the
object cells against the array version, patterns, probabilistic and historical variants,
hexagonal neighbors and rule, and the cyclic CA.
