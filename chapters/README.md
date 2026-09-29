# Chapter notes

Our own notes per book chapter (`NN-<name>.md`): which code implements each example,
design decisions and deviations from the p5.js sketches, the exercises chosen, and how to
run the examples. Notes are added as each chapter is ported (see [ROADMAP.md](../ROADMAP.md)).

| Note | Adds to `noc.common` | Builds on |
|---|---|---|
| [0. Randomness](00-randomness.md) (with the Introduction) | `view` (sketch harness, `Canvas`), `transform`, `tessellate`, `noise`, `mathutils` | — |
| [1. Vectors](01-vectors.md) | `vector` | 0 (walkers, noise) |
| [2. Forces](02-forces.md) | `physics` | 1 (vectors, `Mover`) |
| [3. Oscillation](03-oscillation.md) | — | 2 (`Mover`, forces) |
| [4. Particle systems](04-particle-systems.md) | `view.Canvas.image`, `set_additive`, `make_texture` | 2 (`Mover`), 3 (spaceship) |
| [5. Autonomous agents](05-autonomous-agents.md) | — | 2 (`Mover`), 0 (noise fields) |
| [6. Physics libraries](06-physics-libraries.md) | — | 2 (forces), 3 (springs, pendulums) |
| [7. Cellular automata](07-cellular-automata.md) | `view.Canvas.pixels(smooth=False)` | — |
| [8. Fractals](08-fractals.md) | — | 0 (noise), 1 (vectors) |
| [9. Evolutionary computing](09-evolutionary-computing.md) | — | 2 (`Mover`), 0 (noise) |
