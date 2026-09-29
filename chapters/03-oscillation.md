# Chapter 3 — Oscillation

Run: `uv run python -m noc.ch03_oscillation` (`PageDown`/`PageUp` switch sketches, `H` help)

| # | Sketch | Shows |
|---|---|---|
| 1 | Exercise 3.1 | a baton spinning around its center |
| 2 | Example 3.1 | angular acceleration with `rotate()` |
| 3 | Exercise 3.2 | a baton spun by the mouse and slowed by angular drag |
| 4 | Example 3.2 | movers that spin according to their acceleration |
| 5 | Exercise 3.3 | a cannon: one blast force, then gravity; spinning shells (Space, Up/Down) |
| 6 | Example 3.3 | pointing in the direction of motion |
| 7 | Exercise 3.4 | a car driven with the arrow keys, pointing where it moves |
| 8 | Example 3.4 | polar to Cartesian coordinates |
| 9 | Exercise 3.5 | a spiral |
| 10 | Exercise 3.6 | the Asteroids spaceship (Left/Right, Z) |
| 11 | Example 3.5 | simple harmonic motion from the frame count |
| 12 | Exercise 3.7 | a bob on a "spring" moved by a sine |
| 13 | Example 3.6 | simple harmonic motion with angular velocity |
| 14 | Example 3.7 | oscillator objects |
| 15 | Example 3.8 | a static wave |
| 16 | Example 3.9 | the moving wave |
| 17 | Exercise 3.10 | a wave from Perlin noise |
| 18 | Exercise 3.11 | a `Wave` class, two waves |
| 19 | Exercise 3.12 | additive waves |
| 20 | Example 3.10 | a spring connection (Hooke's law), draggable bob |
| 21 | Exercise 3.14 | a chain of bobs connected by springs |
| 22 | Example 3.11 | the swinging pendulum, draggable |
| 23 | Exercise 3.15 | a double pendulum with its exact equations of motion |
| 24 | Exercise 3.17 | a box sliding down an incline with friction |

## Book sections → code

| Book section | Code |
|---|---|
| Angles; Angular Motion | `ch03_oscillation.oscillation.Spinner`, `RotatingMover`, `Canvas.rotate` |
| Pointing in the Direction of Movement | `Vector.heading` |
| Polar vs. Cartesian Coordinates | `Vector.from_angle` |
| Properties of Oscillation; Oscillation with Angular Velocity | `simple_harmonic`, `Oscillator` |
| Waves | `Wave`, `additive_wave`, `noise_wave` |
| Spring Forces | `Spring` (`force`, `constrain`), `connect` |
| The Pendulum | `Pendulum`, `DoublePendulum` |
| Exercises 3.6, 3.17 | `Spaceship`, `Incline` |

## Design decisions and deviations

- **Springs return forces** (`Spring.force(position)`) instead of applying them
  (`spring.connect(bob)`), like chapter 2's forces; `connect(a, b)` applies equal and opposite
  forces for Exercise 3.14's bob-to-bob springs. Exercise 3.13's `constrainLength()` is
  `Spring.constrain`.
- **The pendulum's angle is measured from straight down** and the bob is
  `pivot + r * (sin θ, cos θ)`, as in the book; dragging sets the angle with a single
  `atan2(dx, dy)` instead of the solution's `atan2(-dy, dx) - 90°`.
- **The double pendulum takes 50 substeps per frame.** The published solution takes one Euler
  step per frame; for this chaotic system the total energy then swings by four times its scale
  within ten seconds (tested). `DoublePendulum.update(substeps)` keeps one frame's worth of
  time but splits it; the sketch shows the energy so the drift is visible. Its trail is kept
  as the last 3000 points instead of a second drawing buffer.
- **Exercise 3.3** uses Example 3.2's rotation rule for the shells; **Exercise 3.4** lets all
  four arrow keys accelerate the car and wraps it around the edges; **Exercise 3.17** reduces
  the box to one dimension along the slope, with static friction holding it while
  `tan(angle) ≤ μ`.
- Not ported: Exercises 3.8, 3.9, 3.13 (fill in the blanks, it's `Spring.constrain`) and 3.16
  (a pen-and-paper question).

## Tests

`tests/ch03_oscillation` covers angular motion, the spaceship, harmonic motion and waves,
Hooke's law (including a damped bob settling at `stretch = weight / k`), the pendulum's
symmetric swing and damping, the double pendulum's energy drift with and without substeps,
and static and sliding friction on the incline.
