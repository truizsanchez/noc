"""Every sketch of every chapter runs a few frames offscreen, with a mouse click, without errors."""

import importlib
import pkgutil

import arcade
import pytest

import noc
from noc.common.view import Sketch, SketchWindow

CHAPTERS = sorted(m.name for m in pkgutil.iter_modules(noc.__path__) if m.name.startswith("ch"))
FRAMES = 20


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_sketches_run(chapter: str) -> None:
    sketches: tuple[type[Sketch], ...] = importlib.import_module(f"noc.{chapter}.sketches").SKETCHES
    window = SketchWindow(sketches, chapter)
    try:
        for index in range(len(sketches)):
            window.show_sketch(index)
            sketch = window.current_view
            assert isinstance(sketch, Sketch)
            x, y = window.width // 2, window.height // 2
            sketch.on_mouse_motion(x, y, 0, 0)
            for frame in range(FRAMES):
                if frame == FRAMES // 4:
                    sketch.on_mouse_press(x, y, arcade.MOUSE_BUTTON_LEFT, 0)
                if frame == FRAMES // 2:
                    sketch.on_mouse_release(x, y, arcade.MOUSE_BUTTON_LEFT, 0)
                sketch.on_fixed_update(1 / 60)
                sketch.on_draw()
    finally:
        window.close()
