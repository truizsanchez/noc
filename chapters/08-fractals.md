# Chapter 8 — Fractals

Run: `uv run python -m noc.ch08_fractals` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1-3 | Examples 8.1-8.3 | recursive circles: nested, two and four children |
| 4 | Exercise 8.1 | a recursive pattern of perpendicular lines |
| 5 | Example 8.4 | the Cantor set |
| 6 | Exercise 8.4 | the Cantor set with objects, one generation per second |
| 7 | Example 8.5 | the Koch curve, five generations |
| 8 | Exercise 8.2 | the Koch snowflake |
| 9 | Exercise 8.3 | the Koch curve drawn from left to right |
| 10 | Exercise 8.5 | the Sierpiński triangle |
| 11 | Example 8.6 | the recursive tree, angle from the mouse |
| 12 | Exercise 8.7 | the same tree with branch thickness by length |
| 13 | Example 8.7 | a stochastic tree, a new one every second |
| 14 | Exercise 8.8 | a tree of branch objects that grows and grows leaves |
| 15 | Exercise 8.9 | a tree whose angles follow Perlin noise, as if in the wind |
| 16 | Example 8.8 | L-system sentences, `A → AB`, `B → A` |
| 17 | Example 8.9 | L-systems drawn by a turtle; `Space` cycles through more (Exercise 8.12) |

## Book sections → code

| Book section | Code |
|---|---|
| Recursion; Implementing Recursive Functions | `ch08_fractals.fractals.nested_circles`, `circles_twice`, `circles_four_times`, `cross_lines` |
| Drawing the Cantor Set with Recursion | `cantor`, `CantorLine`, `cantor_generations` |
| The Koch Curve | `KochLine` (`koch_points`, `replace`), `koch`, `snowflake`, `sierpinski` |
| Trees: deterministic and stochastic | `RecursiveTree.branch` (transformations); `tree`, `stochastic_tree`, `windy_tree`, `GrowingTree` (vectors) |
| L-systems | `ch08_fractals.lsystem.LSystem`, `turtle`, `PRESETS` |

## Design decisions and deviations

- **Fractals as generators.** The book's recursive functions draw as they recurse. Here they
  `yield` what they would draw (`yield from` for the recursive calls), so a fractal is just an
  iterable of circles, segments or branches: the sketches draw it, the tests count and measure
  it, and Exercise 8.3 animates it by drawing a growing prefix.
- **Both ways of building trees.** Example 8.6 keeps the book's transformation stack
  (`canvas.pushed()`, `translate`, `rotate`) to show how `push`/`pop` remember where each
  branch started. The other trees compute positions with vectors, as Exercise 8.8 suggests,
  which is what lets them be generated without drawing.
- **Still pictures are drawn once** on a canvas that keeps them (the book's `noLoop()`):
  Example 8.9's plant has 4,096 segments.
- **Random trees are reproducible**: each is grown from its own `random.Random(seed)`, so the
  stochastic tree stays the same for a whole second and Exercise 8.9's tree keeps its shape
  while the noise moves its angles.
- **L-systems.** Rules are a dict from symbol to replacement, or to weighted alternatives
  (stochastic L-systems, Exercise 8.12). The turtle is a generator using vectors instead of
  transformations (Exercise 8.11) and reports each segment's bracket depth, used to color the
  plants. `PRESETS` adds the Koch curve, the Sierpiński arrowhead, the dragon curve and a
  stochastic plant from *The Algorithmic Beauty of Plants*; drawings are scaled to fit.
- Not ported: Exercises 8.6 (on paper), 8.10 (a spring-physics tree) and 8.13 (audio or text).

## Tests

`tests/ch08_fractals` checks the recursion counts and sizes, the Cantor set's lengths and its
object version against the recursive one, the Koch points (bump pointing up) and total length
(4/3 per generation), the closed snowflake, the Sierpiński count, the binary tree's shape,
reproducible random trees, the growing tree's leaves, L-system generations and stochastic
rules, and the turtle's brackets.
