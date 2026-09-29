# Chapter 2 — Forces

Run: `uv run python -m noc.ch02_forces` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Exercise 2.1 | a helium balloon in a Perlin noise breeze, bouncing off the top |
| 2 | Example 2.1 | gravity, and wind while the mouse is pressed |
| 3 | Example 2.2 | the same forces on a heavy and a light object |
| 4 | Exercise 2.3 | invisible walls: edge forces that grow as objects get closer |
| 5 | Exercise 2.5 | a fan at the mouse blowing the objects away |
| 6 | Example 2.3 | gravity scaled by mass: everything falls alike |
| 7 | Example 2.4 | friction on the floor |
| 8 | Example 2.5 | fluid resistance (drag) in a liquid |
| 9 | Exercise 2.8 | strong drag limited so it can't bounce objects off the surface (`D` toggles) |
| 10 | Example 2.6 | gravitational attraction to a draggable attractor |
| 11 | Example 2.7 | many movers and one attractor |
| 12 | Exercise 2.13 | an attractor that pulls from afar and repels up close |
| 13 | Example 2.8 | two-body attraction |
| 14 | Exercise 2.14 | the figure-eight three-body choreography |
| 15 | Example 2.9 | n bodies attracting one another |
| 16 | Exercise 2.15 | bodies pulled to the mouse and repelling one another |
| 17 | Exercise 2.16 | a spiral galaxy |

## Book sections → code

| Book section | Code |
|---|---|
| Newton's laws; Force Accumulation; Factoring In Mass | `noc.common.physics.Mover` (`apply_force`, `update`) |
| Creating Forces; gravity on Earth | `physics.weight`, `Mover.bounce` |
| Friction | `physics.friction` |
| Air and Fluid Resistance | `physics.drag`, `physics.Liquid` |
| Gravitational Attraction | `physics.attraction` |
| The n-Body Problem | `ch02_forces.forces.attract_all`, `figure_eight`, `Galaxy` |
| Exercises 2.1, 2.3, 2.5, 2.13 | `forces.Balloon`, `edge_repulsion`, `fan`, `attract_and_repel` |

## Design decisions and deviations

- **Forces are functions** that return a vector (`weight(mass)`, `friction(velocity, c)`,
  `drag(velocity, c)`, `attraction(...)`); the book puts them in methods of `Liquid` and
  `Attractor`. `Mover` only knows Newton's second law, so it's reused unchanged by later
  chapters. It lives in `noc.common.physics`; chapter 1's massless `Mover` stays in
  `ch01_vectors`.
- **Exercise 2.2 disappears**: with immutable vectors, `force / self.mass` never touches the
  caller's vector, so there is nothing to `copy()`.
- **One `bounce()`** replaces the per-example `checkEdges()`/`bounceEdges()`: radius,
  restitution (Example 2.4's `-0.9`) and whether there's a ceiling are parameters. Exercise 2.4
  (bounce at the circle's edge, not its center) is the `radius` argument.
- **Simultaneous n-body updates.** Example 2.9 updates body `i` before computing the forces on
  body `i + 1`, so later bodies feel moved neighbors. `attract_all` applies all forces first
  and then the sketches update, so momentum is conserved exactly (tested).
- **Exercise 2.8**: `drag(..., mass=m)` caps the force at `speed * m`, the most that stops the
  object in one frame; the sketch uses a coefficient of 2 with random drop heights, and `D`
  switches the cap off to show the bounce.
- **Exercise 2.14** uses the figure-eight solution of Chenciner and Montgomery (2000). The
  constants are for G = m = 1; `figure_eight()` scales them to pixels and frames, choosing the
  mass so one lap takes 600 frames. Plain Euler steps drift by only a pixel or two per lap
  (tested), without the book's distance clamp, which would break the orbit.
- **Exercise 2.15** has no published solution. A constant pull toward the mouse plus
  inverse-square repulsion and some damping makes the bodies settle into a spaced-out cluster;
  gravity-like attraction to the mouse (weaker when far) flung them off the canvas.
- **Exercise 2.16** follows the published solution, but the 10,000 attractions per frame run
  in numpy (`forces.Galaxy`), one row per star.
- Not ported: Exercises 2.6, 2.7, 2.9–2.12 (variations on the examples shown).

## Tests

`tests/common/test_physics.py` covers Newton's second law, force accumulation, weight, bounces,
friction, drag (and the Exercise 2.8 cap) and the attraction law. `tests/ch02_forces` checks
the balloon, the exercise forces, momentum conservation, the figure eight's return after one
period and the galaxy's stability.
