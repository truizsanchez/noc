# Chapter 6 — Physics Libraries

Run: `uv run python -m noc.ch06_physics` (`PageDown`/`PageUp` switch sketches, `H` help)

The book uses two JavaScript libraries: Matter.js for rigid bodies and Toxiclibs.js for Verlet
particles and springs. Here [pymunk](https://www.pymunk.org/) (Chipmunk2D, already installed
with arcade) replaces Matter.js, and a small Verlet engine of our own replaces Toxiclibs.js,
with pymunk versions of the soft bodies to compare against (`V` switches).

| # | Sketch | Shows |
|---|---|---|
| 1 | Example 6.1 | a body, a ground and a function that draws any pymunk space |
| 2 | Exercise 6.2 | boxes added with the mouse, removed when they leave the canvas |
| 3 | Example 6.3 | falling boxes hitting static boundaries |
| 4 | Example 6.4 | polygon shapes |
| 5 | Example 6.5 | compound bodies: a lollipop of two shapes |
| 6 | Example 6.6 | a pendulum: a `PinJoint` |
| 7 | Exercise 6.5 | a bridge of circles joined by pins, both ends static |
| 8 | Example 6.7 | a windmill: a `PivotJoint` |
| 9 | Exercise 6.7 | the windmill turned by a motor |
| 10 | Example 6.8 | dragging bodies with the mouse |
| 11 | Example 6.9 | attraction, with gravity off |
| 12 | Example 6.10 | collision events: particles change color on contact |
| 13 | Exercise 6.9 | particles that disappear when they collide |
| 14 | Integration interlude | explicit Euler, semi-implicit Euler and Verlet on one orbit |
| 15 | Example 6.11 | a simple Verlet spring |
| 16 | Example 6.12 | a soft string (Verlet or pymunk) |
| 17 | Exercise 6.10 | a hanging cloth (Verlet or pymunk) |
| 18 | Example 6.13 | a soft-body character (Verlet or pymunk) |
| 19 | Example 6.14 | a cluster: a force-directed graph |
| 20 | Exercise 6.13 | eight clusters kept apart by minimum-distance springs |
| 21 | Example 6.15 | attraction and repulsion behaviors |

## Book sections → code

| Book section | Code |
|---|---|
| Matter.js Overview: Engine, Bodies, Render | `pymunk.Space`; `ch06_physics.world` (`new_space`, `add_box`, ...), `sketches.draw_body` |
| Static Bodies; Polygons and Groups of Shapes | `add_box(static=True)`, `add_polygon`, `add_lollipop` |
| Constraints (distance, revolute, mouse) | `pymunk.PinJoint`, `PivotJoint`, `SimpleMotor`, `world.MouseGrab` |
| Adding More Forces | `Body.apply_force_at_world_point` |
| Collision Events | `Space.add_collision_handler`, post-step callbacks |
| A Brief Interlude: Integration Methods | `ch06_physics.integration` |
| Verlet Physics with Toxiclibs.js: world, particles, springs | `ch06_physics.verlet.VerletPhysics` |
| Soft-Body Simulations | `ch06_physics.soft` (`string`, `soft_body`, `cloth`; `VerletModel`, `PymunkModel`) |
| A Force-Directed Graph | `VerletPhysics.add_spring(min_distance=True)`, the cluster sketches |
| Attraction and Repulsion Behaviors | `verlet.Attraction` |

## Design decisions and deviations

- **Matter.js units in pymunk.** Matter integrates 16.67 ms per update and scales gravity by
  0.001; the sketches step pymunk one frame at a time (in 4 substeps), so its default gravity
  becomes 0.28 pixels per frame², its `frictionAir` of 0.01 becomes `space.damping = 0.99`,
  and bodies get Matter's density of 0.001 so masses match. Example 6.9's `G = 0.02` becomes
  5.5 for the same reason.
- **No rigid-body engine of our own.** Collision detection and response between rotating
  polygons is what the chapter says to leave to a library, so the Matter.js part is pymunk
  only; `draw_body` stands in for Matter's renderer by drawing whatever shapes pymunk holds.
- **Exercise 6.7's motor is a `SimpleMotor`**, pymunk's constraint for a constant angular
  velocity, rather than repeated `applyForce` calls. **Exercise 6.9** removes bodies from a
  post-step callback, since pymunk (like Matter) can't remove them mid-step.
- **The integration interlude runs.** The book's "Euler" (`velocity += acceleration;
  position += velocity`) is actually semi-implicit (symplectic) Euler, the variant the book
  credits to Box2D. The sketch puts explicit Euler, semi-implicit Euler and Verlet on the same
  orbit: explicit Euler spirals outward, the other two keep the orbit (tested with its energy).
- **Our Verlet engine follows Toxiclibs's algorithm**: current and previous positions, forces
  from behaviors (gravity, attraction), then 50 relaxation passes over the springs, each moving
  both ends by `strength` of the error split in half (a locked end skips its half), then the
  world bounds. It stores everything in numpy arrays. Toxiclibs relaxes springs one by one
  (Gauss-Seidel); relaxing them all at once would change the method, so the springs are
  colored into groups that share no particle and the groups are relaxed in turn, which is the
  same Gauss-Seidel sweep with each group vectorized.
- **Verlet vs. pymunk soft bodies.** Measured on the three soft bodies (300 frames): both are
  stable; pymunk's `DampedSpring` needed a stiffness tuned per model and 10 substeps per frame,
  then held its shape better (the cloth stretched up to 9% versus Verlet's 59% near the pins,
  where gravity outpulls 50 passes of 0.25-strength relaxation, as in Toxiclibs) and, being
  compiled C, ran faster (cloth: about 5 ms versus 13-16 ms per frame). The sketches show the
  current engine and its time per frame.
- **Dragging** uses `move_to`, which sets both the current and the previous position: the
  book's `lock(); set(); unlock()` idiom, where unlocking clears the velocity.
- **Exercise 6.10's solution starts every row at y = 0** (its `y = y` never advances); the rows
  start 10 pixels apart here.
- **Exercise 6.13** needed drag and an invisible central attractor: with dozens of
  minimum-distance springs per particle, the initial explosion otherwise keeps the clusters
  flying apart. The view zooms to fit the graph, and it relaxes 10 times per frame instead of
  50 (63 spring groups would cost ~57 ms per frame).
- Not ported: Exercises 6.1 (fill in the blank), 6.3, 6.4, 6.6, 6.8, 6.11, 6.12 (design
  variations), 6.14 (combined into Exercise 6.13).

## Tests

`tests/ch06_physics` covers the Verlet engine (integration, gravity, locking, relaxation and
its Toxiclibs weighting, min-distance springs, the spring coloring, momentum, attraction,
bounds), both soft-body engines on all three topologies, the pymunk helpers (landing, compound
bodies, removal, mouse dragging, pin joints) and the energy behavior of the three integrators.
