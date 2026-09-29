# AGENTS.md

Instructions for coding agents (and humans) working on this repo.

Educational, Pythonic port of *The Nature of Code* (2024 p5.js edition), visualized with arcade.
Phases in `ROADMAP.md`, per-chapter notes in `chapters/`. Sister project with the same
conventions: [game-ai](https://github.com/truizsanchez/game-ai).

## Sources (outside the repo, never commit them)
The book's text is CC BY-NC-SA 4.0 and must not be committed. Keep the sources in a sibling
directory; the paths below assume `../nature-of-code-private/`:
- 2024 edition: clone [nature-of-code/noc-book-2](https://github.com/nature-of-code/noc-book-2)
  as `2024_p5js/noc-book-2-main/`. Its p5.js sketches are the **primary reference**:
  `content/examples/<chapter>/<example>/sketch.js`.
- Book as Markdown (one file per chapter; images and screenshots linked into `content/`):
  `uv run tools/html_to_md.py --content ../nature-of-code-private/2024_p5js/noc-book-2-main/content --out ../nature-of-code-private/book-md`.
- 2012 Processing edition, secondary reference (tie-breaker, or an extra example):
  [nature-of-code/noc-examples-processing](https://github.com/nature-of-code/noc-examples-processing)
  under `2012_processing/`.

## Conventions
- Everything in English: code, docstrings, docs.
- Idiomatic Python, not a JavaScript transliteration: dataclasses, Protocols, Enums, generators,
  `match`, type hints. Document deviations from the book in `chapters/NN-*.md`.
- Don't explain Python internals in docs unless asked.
- Simulation logic is pure Python and testable without a window; arcade lives in the sketches
  (`noc.common.view.Sketch`), which draw on a `Canvas` in book coordinates (y down, per frame).
- Shared code goes in `noc.common` only when a chapter needs it.
- One example per `Sketch` subclass, titled like the book ("Example 1.2: ..."), listed in book
  order in the chapter's `SKETCHES`; keep the table in the chapter note in sync.
- Don't name `Sketch` attributes `size`, `center`, `clear`, `width` or `height`: `arcade.View`
  already uses them.

## Commands
- `uv run pytest` · `uv run ruff check` · `uv run ruff format` · `uv run mypy`
- `tests/test_sketches.py` runs every sketch offscreen (`ARCADE_HEADLESS=1`, set in
  `tests/conftest.py`); it needs EGL (on Debian/Ubuntu: `libegl1 libgl1-mesa-dri`).
- Python 3.13 (`.python-version`): pymunk has no 3.14 wheel yet.

## Commits and privacy
- Never write a personal email anywhere (commits, `pyproject.toml`, docs). Commit with the GitHub
  noreply address (repo-local `user.email`); check `git log --format='%ae %ce'` before pushing.
- Maintainer workflow: one branch and PR per change, CI green, then merge locally
  (`git merge --no-ff`, push `main`). Don't use GitHub's merge button or `gh pr merge`: those
  merge commits carry the account's email.
