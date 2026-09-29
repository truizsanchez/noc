# Introduction and Chapter 0 — Randomness

Run: `uv run python -m noc.ch00_randomness` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Introduction, Example #.# | the book's example template: random translucent circles |
| 2 | Example 0.1 | the traditional random walk |
| 3 | Example 0.2 | a uniform distribution as a bar graph |
| 4 | Exercise 0.1 | a walker skewed down and to the right |
| 5 | Example 0.3 | a walker with a 40% chance of moving right |
| 6 | Exercise 0.3 | a walker with a 50% chance of moving toward the mouse |
| 7 | Figure 0.2 | bell curves with a high and a low standard deviation |
| 8 | Example 0.4 | a Gaussian distribution of translucent dots |
| 9 | Exercise 0.4 | paint splatter: Gaussian positions, sizes and colors (keys replace the sliders) |
| 10 | Exercise 0.5 | a Gaussian random walk |
| 11 | Example 0.5 | an accept-reject distribution (probability = x) |
| 12 | Exercise 0.6 | a walk with step sizes from a quadratic accept-reject distribution |
| 13 | Figure 0.4 | Perlin noise versus random values over time |
| 14 | Example 0.6 | the Perlin noise walker |
| 15 | Exercise 0.7 | noise mapped to the step size instead of the position |
| 16 | Figure 0.9 | 2D noise as grayscale pixels |
| 17 | Exercises 0.8, 0.9 | 2D noise with `noiseDetail()` controls, color and a third dimension for animation |
| 18 | Exercise 0.10 | a 3D landscape with noise elevations |

## Book sections → code

| Book section | Code |
|---|---|
| Random Walks; The Random Walker Class | `ch00_randomness.walkers.Walker`, `four_directions` |
| Probability and Nonuniform Distributions | `walkers.tends_right`, `walkers.toward`, `walkers.skewed` |
| A Normal Distribution of Random Numbers | `random.gauss`; `distributions.normal_pdf`, `distributions.Splatter`, `walkers.gaussian` |
| A Custom Distribution of Random Numbers | `distributions.accept_reject`, `distributions.Histogram`, `walkers.quadratic` |
| A Smoother Approach with Perlin Noise; Noise Ranges | `noc.common.noise.Noise` (p5's `noise()`), `noc.common.mathutils.remap` (p5's `map()`), `walkers.NoiseWalker`, `walkers.noise_steps` |
| Two-Dimensional Noise; Noise Detail | `Noise.grid`, `Noise.detail`, `ch00_randomness.landscapes` |

## Design decisions and deviations

- **One walker, many step rules.** The book writes a `Walker` class per variation. Here
  `Walker` holds a *step rule*, a function `(rng, walker) -> (dx, dy)`, and each variation is
  a rule (`four_directions`, `tends_right`, `gaussian(sd)`, ...). Rules that need settings or
  state are built by a factory (`toward(target)`, `noise_steps(noise)`).
- **Random generators are passed in.** Walkers and `accept_reject` take a `random.Random`,
  so the tests seed their own and check the distributions statistically.
- **`noise()` is p5's algorithm, not a library's.** `noc.common.noise` ports p5's value
  noise line by line: the 4096-entry table, cosine interpolation, octaves and falloff
  (`noiseDetail()`), and `noiseSeed()`'s linear congruential generator. A seed gives the same
  table as in p5. `Noise.grid` is a numpy version for whole images; the tests check that it
  matches the scalar one.
- **`map()` is `remap()`**, since `map` is a Python builtin; `constrain()` is `clamp()`.
- **The bell curves use the real normal density.** The book's figure sketch computes
  `e^(-(x - μ)² / σ²)`, missing the factor 2 in the exponent; `normal_pdf` includes it, so the
  low-deviation curve (σ = 0.41) peaks at 0.97 instead of running off the graph.
- **Exercise 0.8's colors match p5's clamping.** The sketch maps noise to a brightness from 0
  to 255 in `colorMode(HSB)`, where brightness tops out at 100, so p5 clamps most pixels to full
  brightness. `landscapes.hue_and_brightness` clamps the same way.
- **Exercise 0.10 without WEBGL.** `landscapes.project` applies the sketch's rotations and
  p5's default perspective camera in numpy, and the quads are drawn back to front (painter's
  algorithm) instead of with a depth buffer. Each quad has one shade instead of p5's per-vertex
  colors.
- **Sliders become keys** in Exercises 0.4 and 0.8/0.9 (listed under `H`).
- The figures of a noise-driven tree and flow field (Figure 0.10) are previews of chapters 8
  and 5 and are ported there. Exercise 0.2 is a pen-and-paper question (its answer is in the
  book).

## Harness notes (`noc.common.view`)

This chapter introduced the pieces every later chapter uses:

- **Book coordinates.** `Canvas` draws with y pointing down and the book's canvas size, scaled
  by `NOC_SCALE`. Motion is in pixels per frame, with one `Sketch.step()` per frame at 60 Hz.
- **Trails.** Sketches that never call `background()` in `draw()` set `background = None`: the
  canvas is a texture that keeps previous frames.
- **Batched drawing.** Each arcade draw call costs ~30 µs, so a few hundred shapes would blow
  the frame budget. `Canvas` turns shapes into triangles (`noc.common.tessellate`: fans, ear
  clipping for concave polygons, mitered strokes) and draws them in one call per frame.
- **Pixels.** `Canvas.pixels(image)` shows a numpy RGB(A) array over the whole canvas
  (p5's `loadPixels()`/`updatePixels()`).

## Tests

`tests/common/test_noise.py` checks the LCG, single-octave values against the table, value
range, smoothness, `detail()` and the numpy grid; `tests/ch00_randomness` checks every step
rule's statistics, accept-reject against its density, the normal density, the splatter
spread, the noise images and the terrain projection.
