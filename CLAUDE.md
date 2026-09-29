# CLAUDE.md

Educational, Pythonic port of *The Nature of Code* (2024 p5.js edition), visualized with arcade.
Phases in `ROADMAP.md`. Sister project with the same conventions: `../game-ai`.

## Sources (outside the repo, never commit them)
- Book as Markdown: `../nature-of-code-private/book-md/` (one file per chapter; images and
  screenshots linked into the source `content/`). Regenerate with `tools/html_to_md.py`.
- 2024 p5.js sketches, **primary reference**:
  `../nature-of-code-private/2024_p5js/noc-book-2-main/content/examples/<chapter>/<example>/sketch.js`.
- 2012 Processing edition, secondary reference: `../nature-of-code-private/2012_processing/`.

## Conventions
- Everything in English: code, docstrings, docs.
- Idiomatic Python, not a JavaScript transliteration: dataclasses, Protocols, Enums, generators,
  `match`, type hints. Document deviations from the book in `chapters/NN-*.md`.
- Don't explain Python internals in docs unless asked.
- Simulation logic is pure Python and testable without a window; arcade lives in the sketches
  (`noc.common.view.Sketch`), which draw on a `Canvas` in book coordinates (y down, per frame).
- Shared code goes in `noc.common` only when a chapter needs it.
- One example per `Sketch` subclass, titled like the book ("Example 1.2: ...").

## Privacy
- Never write the owner's personal email anywhere (commits, `pyproject.toml`, docs). The repo's
  local `user.email` is the GitHub noreply address; check `git log --format='%ae %ce'` before
  pushing.

## Commands
- `uv run pytest` · `uv run ruff check` · `uv run ruff format` · `uv run mypy`
