# Chapter notes

Our own notes per book chapter (`NN-<name>.md`): which code implements each example,
design decisions and deviations from the p5.js sketches, the exercises chosen, and how to
run the examples. [appendices.md](appendices.md) covers the rest of the book (introduction,
creature design, the Ecosystem Project), suggests a route through the port, and lists the
shared modules in `noc.common`.

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
| [10. Neural networks](10-neural-networks.md) | `neural`, `view.Canvas.discs` | 9 (GA for Exercise 10.2) |
| [11. Neuroevolution](11-neuroevolution.md) | `neural.Brains` | 9 (GA), 10 (networks), 5 (steering) |
| [Beyond the chapters](appendices.md) | — | — |

Most notes follow the same order: how to run the sketches, a "Book sections → code" table,
design decisions and deviations from the p5.js sketches (including bugs in the originals that
were fixed), the exercises not ported, and what the tests cover.
