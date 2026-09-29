# noc

A Pythonic, chapter-by-chapter port of Daniel Shiffman's
[*The Nature of Code*](https://natureofcode.com/) (2024 p5.js edition), visualized with
[arcade](https://api.arcade.academy/). The goal is learning: each chapter follows the book's
structure and example numbering, but the code is idiomatic Python rather than a line-by-line
translation of the JavaScript. See [ROADMAP.md](ROADMAP.md) for the phases.

This is an unofficial, non-commercial learning project, not affiliated with the author or the
publisher. The book isn't included; read it for free at [natureofcode.com](https://natureofcode.com/)
or get a copy.

## Setup

```sh
uv sync
uv run pytest
```

## Chapter map

The 2024 p5.js edition is the primary reference; the 2012 Processing edition is only a
tie-breaker or the source of an extra example. "Examples" counts the book's numbered examples.

| Book chapter | Examples | 2024 sketches (`content/examples/`) | 2012 sketches | Python package |
|---|---|---|---|---|
| Introduction | 1 | `00_5_introduction` | `introduction` | `noc.ch00_randomness` |
| 0. Randomness | 6 | `00_randomness` | `introduction` | `noc.ch00_randomness` |
| 1. Vectors | 10 | `01_vectors` | `chp01_vectors` | `noc.ch01_vectors` |
| 2. Forces | 9 | `02_forces` | `chp02_forces` | `noc.ch02_forces` |
| 3. Oscillation | 11 | `03_oscillation` | `chp03_oscillation` | `noc.ch03_oscillation` |
| 4. Particle Systems | 9 | `04_particles` | `chp04_systems` | `noc.ch04_particles` |
| 5. Autonomous Agents | 14 | `05_steering` | `chp06_agents` | `noc.ch05_steering` |
| 6. Physics Libraries | 15 | `06_libraries` | `chp05_physicslibraries` | `noc.ch06_physics` |
| 7. Cellular Automata | 3 | `07_ca` | `chp07_CA` | `noc.ch07_ca` |
| 8. Fractals | 9 | `08_fractals` | `chp08_fractals` | `noc.ch08_fractals` |
| 9. Evolutionary Computing | 5 | `09_ga` | `chp09_ga` | `noc.ch09_ga` |
| 10. Neural Networks | 2 | `10_nn` | `chp10_nn` | `noc.ch10_nn` |
| 11. Neuroevolution | 6 | `11_nn_ga` | — | `noc.ch11_neuroevolution` |

The libraries the book uses are replaced: Matter.js by [pymunk](https://www.pymunk.org/)
(chapter 6, next to our own verlet physics for the Toxiclibs.js part), and ml5.js by small
neural networks written from scratch with numpy (chapters 10 and 11).

## Running the examples

Each chapter opens a window with its examples, in book order: `PageDown`/`PageUp` switch
examples, `H` shows the keys, `P` pauses, `N` steps one frame, `R` restarts. A number
starts at that example: `uv run python -m noc.ch01_vectors 7`.

| Chapter | Command | Shows |
|---|---|---|
| Intro, 0 | `uv run python -m noc.ch00_randomness` | random walks, distributions, Perlin noise |
| 1 | `uv run python -m noc.ch01_vectors` | vector math, motion with velocity and acceleration |
| 2 | `uv run python -m noc.ch02_forces` | gravity, friction, drag, attraction, n bodies |
| 3 | `uv run python -m noc.ch03_oscillation` | angular motion, waves, springs, pendulums |
| 4 | `uv run python -m noc.ch04_particles` | emitters, forces on systems, textures, additive blending |
| 5 | `uv run python -m noc.ch05_steering` | steering, flow fields, path following, flocking |
| 6 | `uv run python -m noc.ch06_physics` | pymunk bodies and joints, Verlet springs and soft bodies |
| 7 | `uv run python -m noc.ch07_ca` | elementary CA, the Game of Life, hexagonal and cyclic CA |
| 8 | `uv run python -m noc.ch08_fractals` | recursion, Koch curves, trees, L-systems |
| 9 | `uv run python -m noc.ch09_ga` | genetic algorithms, smart rockets, interactive selection, ecosystem |
| 10 | `uv run python -m noc.ch10_nn` | the perceptron, XOR, a gesture classifier |
| 11 | `uv run python -m noc.ch11_neuroevolution` | Flappy Bird, neuroevolving rockets and creatures, an ecosystem |

The canvas keeps the book's coordinates (usually 640x240, y pointing down, units per frame)
and is shown scaled up; set `NOC_SCALE` to change the factor (default 2).

## Repository layout

- `src/noc/common/`: code shared by several chapters. `view.py` is the arcade harness
  (`Sketch`, and `Canvas` for p5-style drawing); everything else is pure Python.
- `src/noc/chNN_*/`: one package per chapter, with the simulation separate from the sketches
  that draw it.
- `chapters/`: our notes per chapter: examples mapped to code, deviations from the JavaScript
  and the exercises chosen. Start at [chapters/README.md](chapters/README.md).
- `tests/`: mirrors `src/noc`.
- `tools/`: the book-to-Markdown converter.

Checks: `uv run pytest`, `uv run ruff check`, `uv run ruff format`, `uv run mypy`.

## Book text

The book's text is under CC BY-NC-SA 4.0 and is **not** part of this repo.
`tools/html_to_md.py` converts a local copy of the book's source repository
([nature-of-code/noc-book-2](https://github.com/nature-of-code/noc-book-2)) into Markdown in
a directory outside the repo:

```sh
uv run tools/html_to_md.py \
    --content ../nature-of-code-private/2024_p5js/noc-book-2-main/content \
    --out ../nature-of-code-private/book-md
```

Images and example screenshots are linked in place from `content/`, not copied.

## License

The code in this repository is under the [MIT License](LICENSE). The book, its text and
figures, and the original sketches remain the property of their copyright holders and are not
covered by it.
