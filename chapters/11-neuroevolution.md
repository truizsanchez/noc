# Chapter 11 — Neuroevolution

Run: `uv run python -m noc.ch11_neuroevolution` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Example 11.1 | a playable Flappy Bird clone, with a score (Exercise 11.1) |
| 2 | Example 11.2 | 200 birds whose brains evolve; speed-up keys and a status overlay (Exercise 11.2) |
| 3 | Example 11.3 | smart rockets whose brain picks the thrust from their position; `V` adds velocity inputs (Exercise 11.4) |
| 4 | Example 11.4 | creatures evolving to chase a moving glow; `C` switches the force encoding |
| 5 | Example 11.5 | a bloop with 15 whisker sensors (move it with the mouse) |
| 6 | Example 11.6 | an ecosystem of sensing bloops that eat, starve and reproduce |

`1`, `2`, `3` run 1, 10 or 100 simulation steps per frame in the evolving sketches.

## Book sections → code

| Book section | Code |
|---|---|
| Coding Flappy Bird | `ch11_neuroevolution.games.Bird`, `Pipe`, `Course` |
| The Bird Brain; A Flock of Flappy Birds; Selection; Heredity | `Flock`; `noc.common.neural.Brains` (`predict`, `next_generation`) |
| Steering the Neuroevolutionary Way | `RocketBrains`, `steering_force` |
| Responding to Change; Speeding Up Time | `Glow`, `Seekers`; `SpeedSketch` |
| A Neuroevolutionary Ecosystem: sensing, learning from the sensors | `sensor_offsets`, `sense`, `Ecosystem` |

## Design decisions and deviations

- **A population of brains in one array.** ml5.js gives each bird its own network and the
  sketch asks each one to think, which in Python would be hundreds of small numpy calls per
  frame. `Brains` stacks every network's weights along a first axis, so one `einsum` per
  layer evaluates the whole population, and selection, crossover and mutation work on the
  stacks. `member(i)` extracts an ordinary `NeuralNetwork` (whose `to_json`/`from_json` cover
  Exercise 11.3's saving and loading); a test checks the stack predicts exactly what its
  members do. The agents' positions and velocities are arrays too, so a simulation step costs
  about 0.1 ms and 100 steps per frame (Exercise 11.2) are affordable.
- **Selection** draws parents with `rng.choice(p=fitness / total)`, which gives the same odds
  as the book's relay-race `weightedSelection()`; crossover takes each weight from either parent
  and mutation adds Gaussian noise to about `rate` of the weights, like ml5's neuroevolution.
- **Brains**: 16 sigmoid hidden neurons; Flappy Bird's outputs are a softmax over
  flap / no flap (ml5's classification), the steering brains' are two sigmoids read as an
  angle (×2π) and a magnitude, as in the book.
- **Example 11.4's force encoding.** With angle and magnitude, an untrained network's outputs
  sit near 0.5, so everyone pushes toward angle π (left), and turning toward a target means
  learning a mapping that wraps around at 2π. `C` switches to outputs read as the force's x
  and y (the Reynolds-style idea behind Exercise 11.4); measured over 60 generations with three
  seeds, time spent on the glow grew from about 60 to about 240 with the book's encoding and
  to about 4,600 with components.
- **The ecosystem never goes extinct**: with the book's parameters a population of 20 random
  brains often dies out within a few thousand steps, which ends the demonstration. When fewer
  than two bloops remain, newcomers with random brains arrive.
- The Flappy course keeps only the newest pipe between generations, as the solution does.

## Tests

`tests/ch11_neuroevolution` checks the bird's physics against the book's order of operations,
pipe collisions and scoring, that neuroevolution produces birds that live several times longer
than the first generation's best, both force encodings, rockets stopping at obstacles and
starting a new generation, seekers scoring on the glow, the whisker sensors, and the
ecosystem's bookkeeping. `tests/common/test_neural.py` checks `Brains` against its members and
its selection.
