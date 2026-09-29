# Chapter 5 — Autonomous Agents

Run: `uv run python -m noc.ch05_steering` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Example 5.1 | seeking the mouse |
| 2 | Exercise 5.1 | fleeing from the mouse |
| 3 | Exercise 5.3 | pursue and evade, with the predicted position |
| 4 | Example 5.2 | arriving: slowing down near the mouse |
| 5 | Exercise 5.4 | Reynolds's wander, with its circle (click to hide) |
| 6 | Example 5.3 | staying within walls |
| 7 | Example 5.4 | flow-field following; swirl (Exercise 5.6) and animated noise (Exercise 5.7) |
| 8 | Exercise 5.9 | the angle between two vectors |
| 9 | Example 5.5 | a path with a radius |
| 10 | Example 5.6 | simple path following, with the debug drawing |
| 11 | Example 5.7 | a path of several segments |
| 12 | Example 5.8 | path following on a random path (click for another) |
| 13 | Example 5.9 | separation (drag to add vehicles) |
| 14 | Example 5.10 | seek and separate, weighted |
| 15 | Exercise 5.13 | crowd path following: separation plus a closed path |
| 16 | Example 5.11 | flocking; seek the mouse (Exercise 5.16) and weights on keys (Exercise 5.18) |
| 17 | Example 5.12 | flocking with a bin lattice, counting neighbor checks (`D` shows the grid) |
| 18 | Example 5.13 | flocking with a quadtree (`D` shows the nodes) |
| 19 | Example 5.14 | a sine/cosine lookup table |

## Book sections → code

| Book section | Code |
|---|---|
| Vehicles and Steering; The Steering Force | `ch05_steering.vehicle.Vehicle` (`steer_toward`), `seek`, `flee` |
| The Arrive Behavior; Your Own Behaviors | `arrive`, `pursue`, `evade`, `Wander`, `stay_within` |
| Flow Fields | `FlowField` (`from_noise`, `swirl`, `lookup`), `follow_field` |
| The Dot Product; Path Following | `normal_point`, `closest_on_segment`, `Path`, `follow_path` |
| Implementing Group Behaviors; Combining Behaviors | `separate`, `align`, `cohere`, weighted sums in the sketches |
| Flocking | `flock` (objects), `ch05_steering.flocking.Flock` (numpy) |
| Algorithmic Efficiency; Spatial Subdivisions | `BinLattice`, `QuadTree`, `Rect` |
| More Optimization Tricks | `SinCosTable`; `mag_sq` in `Vector.limit` |

## Design decisions and deviations

- **Behaviors return forces.** The book mixes both styles (`seek()` applies the force in
  Example 5.1 and returns it in Example 5.10); here every behavior returns a force already
  limited to `max_force`, so combining behaviors is a weighted sum. `Vehicle` is chapter 2's
  `Mover` with `max_speed` (its `top_speed`), `max_force` and a size.
- **General path segments.** The book decides whether the normal point lies on a segment by
  comparing x-coordinates, which only works for paths running left to right. `follow_path`
  clamps the projection to the segment instead (Exercise 5.10's question), so it also handles
  Exercise 5.13's closed loop.
- **Flocking, twice.** `flock()` works on `Vehicle` objects, like the book, in a single pass
  over the neighbors (the test checks it equals `separate + align + cohere`). The plain
  flocking sketch uses `Flock`, which holds the flock in numpy arrays and computes all `n²`
  distances at once: the tests check it gives the same forces as the object version. Numpy
  updates all boids simultaneously, whereas the book updates each boid before computing the
  next one's forces.
- **Spatial subdivision sketches show the savings.** Examples 5.12 and 5.13 keep boids as
  objects and report the neighbor checks made against the `n²` a brute-force scan needs.
  The quadtree is rebuilt every frame from the boids (the book's Example 5.13 only
  demonstrates insertion and queries on points).
- **Flow fields store angles** in a numpy array; noise fields come from `Noise.grid`.
- Not ported: Exercises 5.2, 5.5, 5.8, 5.11, 5.14, 5.15, 5.17, 5.19, 5.21 (open-ended or
  variations); Exercise 5.12 is `cohere`; Exercise 5.20 is Example 5.13.

## Tests

`tests/ch05_steering` checks each behavior's force, a vehicle arriving at rest, fields and
their lookup, path geometry and path following (open and closed), the group behaviors, the
single-pass and numpy flocks against the per-behavior functions, and the bin lattice and
quadtree against brute-force scans.
