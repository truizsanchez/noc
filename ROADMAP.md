# Roadmap: "The Nature of Code" → idiomatic Python (arcade)

## Context
New, empty repo `truizsanchez/noc` (only a PyCharm `main.py` and `.idea/`). Goal: an educational,
chapter-by-chapter port of Daniel Shiffman's *The Nature of Code* to **Pythonic** code, visualized
with arcade, analogous to `../game-ai` (same stack, conventions, working method, phase structure).

Source material (`../nature-of-code-private/`, outside the repo, never committed):
- `2024_p5js/noc-book-2-main/content/` — **2024 p5.js edition, primary reference**: one HTML file per
  chapter (`00_randomness.html` … `11_nn_ga.html`), `images/`, and `examples/<chapter>/<example>/sketch.js`
  (~165 sketches, examples + exercise solutions). Book text is CC BY-NC-SA 4.0.
- `2012_processing/` — 2012 Processing edition (book HTML in `noc-book-master/chapters/`, Java examples in
  `noc-examples-processing-master/`, MIT). Secondary: tie-breaker, or source of an extra example worth porting.

Decisions taken:
- 2024 edition drives chapter structure and example numbering; 2012 only when it adds something.
- Scope per chapter: every numbered example + a selection of key exercises (those that add a concept, e.g. cloth, bridge).
- Ch. 10–11 neural networks: **numpy from scratch** (perceptron, MLP with backprop, neuroevolution); no ml5/TF equivalent.
- Ch. 6 physics libraries: **stop at the start of that phase to decide together**. Intent: pymunk replaces Matter.js;
  for the Toxiclibs part (verlet particles, springs, soft bodies) check whether pymunk (or another library) covers it,
  and if so show **both** our own verlet implementation and the library version side by side.
- Everything in English (code, docs, docstrings); don't explain Python internals unless asked.
- Book Markdown lives outside the repo (`../nature-of-code-private/book-md/`).

## Repo layout (target)
```
noc/
  pyproject.toml          # uv, src layout, Python >=3.13, arcade (+ numpy from ch.10, pymunk from ch.6)
  src/noc/
    common/               # shared, extracted only when a chapter needs it: Vector, noise (Perlin, p5-compatible),
                          # random helpers (gaussian, accept-reject, custom distributions), map/constrain, demo harness
    ch00_randomness/ ch01_vectors/ ch02_forces/ ch03_oscillation/ ch04_particles/ ch05_steering/
    ch06_physics/ ch07_ca/ ch08_fractals/ ch09_ga/ ch10_nn/ ch11_neuroevolution/
  chapters/NN-<name>.md   # per chapter: book sections/examples → code, deviations from the JS, exercises chosen
  tools/html_to_md.py     # book HTML → Markdown (output outside the repo)
  tests/                  # mirrors src/noc
```
Per-chapter pattern: pure simulation logic (testable without a window) + thin arcade view; `python -m noc.chNN_*`
opens a demo menu where each book example is one numbered demo (`1`–`9`/page keys, `H` help, `P` pause, `N` step).
Canvas default 640×240 like the book's sketches. Stack: uv, ruff (same rule set), mypy strict, pytest, GitHub Actions CI.

## Phases
Progress: `[x]` merged into `main`, `[ ]` pending.

**[x] Phase 0 — Bootstrap + book to Markdown**
- Remove PyCharm `main.py`; ignore `.idea/`. pyproject/ruff/mypy/pytest config copied from `../game-ai`, CI workflow,
  README (chapter map table: book chapter ↔ 2024 example dirs ↔ 2012 dirs ↔ package), CLAUDE.md, LICENSE (MIT, code only).
- `tools/html_to_md.py`: 2024 content HTML → one `.md` per chapter + images into `../nature-of-code-private/book-md/`;
  code blocks as fenced JS, math kept readable (the HTML uses `data-type="equation"` / KaTeX).
- Demo harness in `noc.common.view` (sketch-like: `setup`/`step`/`draw`, fixed timestep, help overlay, demo switching),
  modelled on `../game-ai/src/gameai/common/view.py`.

**[x] Phase 1 — Introduction + Ch.0 Randomness**
- Random walkers, distributions, gaussian, accept-reject, custom probability, **Perlin noise** (port of p5's `noise()`
  so results match the book, with octaves/falloff), 1D/2D noise examples.

**[x] Phase 2 — Ch.1 Vectors** — `Vector` (decide: own dataclass vs. `pyglet.math.Vec2`; p5-like API is mutable,
  Python version likely immutable with operators), bouncing ball, Mover, acceleration towards the mouse.

**[x] Phase 3 — Ch.2 Forces** — Newton's laws, mass, gravity/wind, friction, drag (liquid), gravitational attraction, n-body.

**[x] Phase 4 — Ch.3 Oscillation** — angles, angular motion, polar coordinates, simple harmonic motion, waves,
  pendulum, springs.

**[x] Phase 5 — Ch.4 Particle Systems** — particle/emitter, many emitters, inheritance & polymorphism (→ Protocols),
  forces and repellers on systems, image textures and additive blending.

**[x] Phase 6 — Ch.5 Autonomous Agents** — seek/arrive, desired velocity, flow fields, path following,
  separation/alignment/cohesion, flocking, spatial subdivision. (Kept independent from game-ai; Shiffman's
  Reynolds-style formulation differs from Buckland's.)

**[ ] Phase 7 — Ch.6 Physics Libraries** — ⚠ **decision point before coding**: survey pymunk (and alternatives) for
  both halves; then Matter.js examples → pymunk (bodies, static bodies, polygons/compound, constraints, mouse,
  attraction, collision events); integration-methods interlude (Euler vs. verlet); Toxiclibs examples → own verlet
  physics **and** library version where one exists (springs, soft string, soft body, force-directed graph, attraction).

**[ ] Phase 8 — Ch.7 Cellular Automata** — elementary CA (Wolfram rules), Game of Life (numpy optional), object-oriented
  cells, variations (hexagonal/probabilistic/continuous as exercises).

**[ ] Phase 9 — Ch.8 Fractals** — recursion, Cantor set, Koch curve, recursive trees, L-systems.

**[x] Phase 10 — Ch.9 Evolutionary Computing** — GA for Shakespeare monkey, fitness/selection/crossover/mutation,
  smart rockets, interactive selection, evolving ecosystem.

**[x] Phase 11 — Ch.10 Neural Networks** — perceptron (with normalization), multilayer network with backprop in numpy,
  gesture classifier recreated with the numpy MLP (mouse-drawn training data).

**[x] Phase 12 — Ch.11 Neuroevolution** — Flappy Bird, neuroevolutionary Flappy Bird, neuroevolution smart rockets,
  steering, creature sensors, ecosystem.

**[x] Phase 13 (optional) — Wrap-up docs** — chapters index, README demo table, Creature Design / resources notes.

## Working method (every chapter phase, as in game-ai)
1. Read the chapter Markdown + 2024 sketches (2012 only as tie-breaker); write `chapters/NN-*.md` outline mirroring the book.
2. Implement logic per section with tests (e.g. noise values vs. p5, forces, CA rule outputs, GA operators, perceptron
   convergence); note Pythonic deviations from the JS explicitly; list the exercises chosen.
3. arcade demo; ruff + format + mypy + pytest green; feature branch → PR → CI green → merge → next phase.

## Privacy: no real email anywhere
The global git config uses a personal address; the repo must never inherit it.
Before the **first commit**: set the repo-local `user.email` to the GitHub noreply address
(as in `../game-ai`). `pyproject.toml` authors get only
`{ name = "truizsanchez" }`, no email; nothing in README/LICENSE/docs contains it. Before each push, check
`git log --format='%ae %ce'` shows only the noreply address. Record this in CLAUDE.md and memory.

## Verification
- Phase 0: all chapters in `book-md/` with readable code, equations and images; checks pass on the skeleton; CI green.
- Chapter phases: unit tests on pure logic + running `uv run python -m noc.chNN_*` (offscreen screenshots) and
  comparing with the book's screenshots in `content/examples/*/screenshot.png`.
