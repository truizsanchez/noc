# Chapter 4 — Particle Systems

Run: `uv run python -m noc.ch04_particles` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Example 4.1 | a single particle, started over when it dies |
| 2 | Example 4.2 | an array of particles, removing the dead ones |
| 3 | Example 4.3 | a particle emitter |
| 4 | Exercise 4.3 | an emitter that follows the mouse |
| 5 | Exercise 4.4 | chapter 3's spaceship with thruster particles (Left/Right, Up) |
| 6 | Example 4.4 | a system of systems: click to add emitters |
| 7 | Exercise 4.5 | emitters of 100 particles, removed once empty |
| 8 | Exercise 4.6 | blocks that shatter into particles when clicked |
| 9 | Example 4.5 | inheritance and polymorphism: circles and spinning confetti |
| 10 | Example 4.6 | a particle system with forces (with trails) |
| 11 | Example 4.7 | a particle system with a repeller |
| 12 | Exercise 4.9 | several attractors and repellers |
| 13 | Example 4.8 | an image-texture smoke system, blown by the mouse |
| 14 | Exercise 4.11 | fire: additive blending, yellow to red |
| 15 | Exercise 4.12 | an array of textures, one picked per particle |
| 16 | Example 4.9 | additive blending |

## Book sections → code

| Book section | Code |
|---|---|
| A Single Particle | `ch04_particles.particles.Particle` (a `noc.common.physics.Mover` with a lifespan) |
| An Array of Particles; A Particle Emitter | `Emitter` (`add_particle`, `run`) |
| A System of Emitters; Exercise 4.5 | a list of emitters; `Emitter.budget`, `finished` |
| Inheritance and Polymorphism | `Confetti(Particle)`; `sketches.draw_particle` matches on the class |
| Particle Systems with Forces / Repellers | `Emitter.apply_force`, `Emitter.apply`, `PointForce` |
| Image Textures and Additive Blending | `ch04_particles.textures`, `Canvas.image`, `Canvas.set_additive` |
| Exercise 4.6 | `Shard`, `block`, `shatter` |

## Design decisions and deviations

- **`Particle` extends chapter 2's `Mover`** (it's a dataclass subclass adding `lifespan` and
  `decay`), so forces, mass and `apply_force` come for free. The lifespan doubles as opacity
  (`alpha`).
- **Polymorphism without drawing methods.** The book's `Confetti` overrides `show()`. Here the
  particles hold no drawing code, so `Confetti` adds nothing but its type, and the sketch
  dispatches with structural pattern matching (`case Confetti(position=...)`). Behavior that
  does differ lives in subclasses too: `Shard` adds drag, `TexturedParticle` remembers its
  texture.
- **Particle factories.** An `Emitter` takes a function `(origin, rng) -> Particle` instead
  of hard-coding `new Particle(...)`, so the same class emits falling particles, confetti,
  smoke, exhaust or sparks.
- **Removing the dead**: `run()` rebuilds the list with a comprehension, which is the
  Pythonic version of the book's backward loop with `splice()` (and its `filter()` variant).
- **Exercise 4.9**: one `PointForce` class whose sign decides repel or attract, instead of
  separate `Repeller` and `Attractor` classes.
- **Textures are generated** with numpy (`textures.blob`, `ring`, `star`), as the book
  suggests is possible, instead of shipping the book's PNG files.
- **Additive blending** weights by alpha (`SRC_ALPHA, ONE`), like p5's `blendMode(ADD)`; arcade's
  own additive mode adds a texture's color even where it's transparent. The harness learned
  `Canvas.image()` (sprites drawn in one batch) and `Canvas.set_additive()` for this chapter.
- **Faster batching.** A thousand particles exposed the cost of building vertices in Python:
  the shape batch now collects positions in a flat list and colors as runs expanded with
  numpy, and circles use cached sine tables and a direct ring for their outline (1270 stroked
  circles: 49 → 21 ms per frame).
- Not ported: Exercises 4.1, 4.2, 4.7, 4.8 (small variations), 4.10 (all-pairs interaction is
  chapter 5's topic), 4.13 (it's `add_particle(count)`), 4.14 (p5's other blend modes).

## Tests

`tests/ch04_particles` covers lifespan and fading, emitter populations and budgets, forces on
systems, the repeller/attractor law, shattering, and the generated textures' alpha.
