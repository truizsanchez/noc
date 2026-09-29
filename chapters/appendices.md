# Beyond the chapters

The book's front and back matter, and the running project, mapped to this port.

| Part of the book | In this port |
|---|---|
| Introduction (how to read the code, the example template) | Chapter 0's first sketch, "Example #.#", is the template; `chapters/00-randomness.md` describes the harness that plays the role of `setup()`/`draw()` |
| The Ecosystem Project (a prompt at the end of every chapter) | Not ported as such: it's an open invitation to build your own world. Examples 9.5 (evolving bloops) and 11.6 (bloops with brains) are the book's own versions of it |
| Appendix: Creature Design | Drawing advice (paper first, nine basic shapes), no code. The `Canvas` API covers the ingredients: `circle`, `ellipse`, `rect`, `polygon`, `polyline`, `line`, `point`, transformations |
| Additional Resources | The book's reading list; the Coding Train videos linked in each chapter are the same for this port |
| Credits and image credits | Not applicable: no figures are included |

## Using the port as a course

Each chapter is one package, `noc.chNN_*`, with the simulation in plain modules and the
sketches in `sketches.py`. A reasonable route through it:

1. Run the chapter (`uv run python -m noc.chNN_...`) and page through its sketches alongside
   the book.
2. Read the chapter note (`chapters/NN-*.md`): its "Book sections → code" table points at the
   code for each section, and "Design decisions and deviations" explains where the Python
   differs from the JavaScript and why.
3. Read the tests (`tests/chNN_*`): many reproduce the book's claims numerically (a glider
   crossing the board, a Koch curve growing by 4/3, momentum conserved in the n-body
   simulation, a perceptron learning a line).
4. Try the exercises that weren't ported (each note lists them) as additions to the chapter's
   `sketches.py`.

## Shared building blocks (`noc.common`)

| Module | What it is | First used |
|---|---|---|
| `view` | the arcade harness: `Sketch`, `Canvas` (p5-style drawing in book coordinates, batched), `run` | 0 |
| `transform` | p5's `translate`/`rotate`/`scale` stack as a value | 0 |
| `tessellate` | shapes to triangles for the batched renderer | 0 |
| `noise` | p5's `noise()`, value noise with octaves and seeds, plus a numpy grid version | 0 |
| `mathutils` | `remap` (p5's `map()`), `clamp` | 0 |
| `vector` | an immutable 2D vector with `p5.Vector`'s method names | 1 |
| `physics` | a `Mover` with mass and forces, and chapter 2's forces | 2 |
| `neural` | a numpy neural network, an ml5-like classifier, and stacked populations for neuroevolution | 10 |
