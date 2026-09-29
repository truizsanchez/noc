# Chapter 1 — Vectors

Run: `uv run python -m noc.ch01_vectors` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Example 1.1 | a bouncing ball with separate `x`/`y` variables |
| 2 | Example 1.2 | the same ball with position and velocity vectors |
| 3 | Exercise 1.1 | chapter 0's walker with a random unit vector as its step |
| 4 | Example 1.3 | vector subtraction: mouse minus center |
| 5 | Example 1.4 | multiplying a vector by 0.5 |
| 6 | Example 1.5 | a vector's magnitude as a bar |
| 7 | Example 1.6 | the normalized vector, scaled to 50 |
| 8 | Example 1.7 | Motion 101: position + velocity, wrapping at the edges |
| 9 | Example 1.8 | constant acceleration with a top speed |
| 10 | Exercise 1.5 | a train that accelerates (Up) and brakes (Down) |
| 11 | Example 1.9 | random acceleration |
| 12 | Exercise 1.6 | acceleration from Perlin noise |
| 13 | Example 1.10 | accelerating toward the mouse |
| 14 | Exercise 1.8 | acceleration toward the mouse, stronger with distance |

## Book sections → code

| Book section | Code |
|---|---|
| Vectors in p5.js; Vector Addition; More Vector Math | `noc.common.vector.Vector` (`+`, `-`, `*`, `/`) |
| Vector Magnitude; Normalizing Vectors | `Vector.mag`, `mag_sq`, `normalize`, `set_mag` |
| Motion with Vectors | `ch01_vectors.motion.Mover`, `wrap_edges`, `bounce_edges` |
| Acceleration (constant, random, interactive); Exercise 1.4's `limit()` | `Mover.update`, `Vector.limit`, `random_acceleration`, `toward`, `toward_by_distance` |
| Static vs. Nonstatic Methods; Exercise 1.7 | not needed: see below (`tests/common/test_vector.py::test_exercise_1_7`) |

## Design decisions and deviations

- **Immutable vectors with operators.** `p5.Vector` methods modify the vector they're called
  on, which is why the book needs static versions (`p5.Vector.sub(a, b)`) and spends a
  section on the difference. `Vector` is a frozen dataclass: `a - b` returns a new vector,
  `position += velocity` rebinds the name, and a vector shared between two objects can't be
  changed through one of them. The module docstring has the p5-to-Python table.
- **`Vector` is our own class**, not `pyglet.math.Vec2`: the method names follow `p5.Vector`
  (`mag`, `limit`, `set_mag`, `heading`, `from_angle`, `random2d`) so the book's code maps
  one to one. `v.xy` gives a `(x, y)` tuple for drawing calls: `canvas.circle(*v.xy, 48)`.
- **One `Mover`, acceleration rules as functions**, as with chapter 0's walkers: the examples
  only differ in how `acceleration` is set each frame.
- **Exercise 1.8's solution has a bug**: it limits the velocity with `this.topspeed`, which
  is undefined (the field is `topSpeed`), so p5's `limit()` does nothing. Here the top speed of
  5 applies.
- **Exercise 1.5** draws a small locomotive instead of loading the solution's image; the train
  can brake to a stop but not reverse, as in the solution.
- Not ported: Exercise 1.2 (open-ended) and Exercise 1.3 (a 3D bouncing ball with
  `orbitControl()`; 3D is outside the book's focus and would need a 3D camera).
  Exercise 1.6 has no published solution; ours maps two noise offsets to the acceleration's
  axes.

## Tests

`tests/common/test_vector.py` covers the vector API, including Exercise 1.7's pseudocode;
`tests/ch01_vectors/test_motion.py` covers the motion algorithm, top speed, edge wrapping and
bouncing, and each acceleration rule.
