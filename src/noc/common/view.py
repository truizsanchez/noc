"""Minimal arcade harness shared by every chapter's examples.

The book's examples are p5.js sketches: a small canvas (usually 640x240) with y pointing
down, a ``draw()`` called once per frame, and motion measured in pixels per frame. A
:class:`Sketch` keeps all three so the numbers in the book carry over unchanged:

- :meth:`Sketch.step` advances the simulation by one frame, at a fixed 60 Hz (arcade's
  ``on_fixed_update``), independent of the rendering rate.
- :meth:`Sketch.draw` receives a :class:`Canvas` that works in book coordinates (origin at the
  top left, y down, sizes as diameters) and is shown scaled up (``NOC_SCALE``, default 2).
- ``background = None`` keeps what was drawn in previous frames, for sketches that never
  call ``background()`` in ``draw()`` (random walkers, trails).

Common keys: ``P`` pause, ``N`` one step while paused, ``R`` restart, ``H`` help,
``PageDown``/``PageUp`` next/previous example.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import ClassVar

import arcade
from arcade.gl import geometry
from arcade.types import LRBT

from noc.common.transform import IDENTITY

type Color = tuple[int, int, int] | tuple[int, int, int, int]
type Point = tuple[float, float]

FIXED_RATE = 1 / 60
SCALE = float(os.environ.get("NOC_SCALE", "2"))

BLACK: Color = (0, 0, 0)
WHITE: Color = (255, 255, 255)


def gray(value: int, alpha: int = 255) -> Color:
    """p5's ``fill(value)`` / ``fill(value, alpha)``."""
    return (value, value, value, alpha)


class Canvas:
    """p5-style drawing in book coordinates, with explicit colors instead of global state.

    ``fill=None`` is ``noFill()`` and ``stroke=None`` is ``noStroke()``; the defaults are
    p5's (white fill, black 1-pixel stroke).
    """

    def __init__(self, window: arcade.Window, width: int, height: int, *, persistent: bool) -> None:
        self.width, self.height = width, height
        self.transform = IDENTITY
        pixels = (round(width * SCALE), round(height * SCALE))
        viewport = LRBT(0, pixels[0], 0, pixels[1])
        center = (width / 2, height / 2)
        # y-down camera for shapes; text is drawn with a y-up one so glyphs aren't mirrored.
        self._camera = arcade.Camera2D(
            viewport=viewport,
            position=center,
            projection=LRBT(-width / 2, width / 2, height / 2, -height / 2),
            window=window,
        )
        self._text_camera = arcade.Camera2D(
            viewport=viewport,
            position=center,
            projection=LRBT(-width / 2, width / 2, -height / 2, height / 2),
            window=window,
        )
        self._texts: dict[tuple[str, float, Color, str, str], arcade.Text] = {}
        self._ctx = window.ctx
        self._fbo = None
        if persistent:
            self._texture = self._ctx.texture(pixels, components=4)
            self._fbo = self._ctx.framebuffer(color_attachments=[self._texture])
            self._fbo.clear(color=(*WHITE, 255))
            self._quad = geometry.quad_2d_fs()

    @contextmanager
    def frame(self) -> Iterator[None]:
        """Draw one frame; on a persistent canvas, into the texture that keeps the old ones."""
        self.transform = IDENTITY
        if self._fbo is None:
            with self._camera.activate():
                yield
            return
        with self._fbo.activate(), self._camera.activate():
            yield
        # Copy without blending: the texture's alpha is below 1 wherever translucent shapes
        # were drawn, and blending it over the window would wash those colors out.
        self._texture.use(0)
        with self._ctx.enabled_only():
            self._quad.render(self._ctx.utility_textured_quad_program)

    # --- transformations (p5's push/pop, translate, rotate, scale) -------------------
    @contextmanager
    def pushed(self) -> Iterator[None]:
        saved = self.transform
        try:
            yield
        finally:
            self.transform = saved

    def translate(self, x: float, y: float) -> None:
        self.transform = self.transform.translated(x, y)

    def rotate(self, angle: float) -> None:
        self.transform = self.transform.rotated(angle)

    def scale(self, factor: float) -> None:
        self.transform = self.transform.scaled(factor)

    # --- shapes -------------------------------------------------------------------------
    def background(self, color: Color) -> None:
        arcade.draw_lrbt_rectangle_filled(0, self.width, 0, self.height, color)

    def line(
        self, x1: float, y1: float, x2: float, y2: float, stroke: Color = BLACK, weight: float = 1
    ) -> None:
        (ax, ay), (bx, by) = self.transform.apply(x1, y1), self.transform.apply(x2, y2)
        arcade.draw_line(ax, ay, bx, by, stroke, weight * self.transform.scale)

    def point(self, x: float, y: float, stroke: Color = BLACK, weight: float = 1) -> None:
        px, py = self.transform.apply(x, y)
        arcade.draw_point(px, py, stroke, weight * self.transform.scale)

    def circle(
        self,
        x: float,
        y: float,
        diameter: float,
        fill: Color | None = WHITE,
        stroke: Color | None = BLACK,
        weight: float = 1,
    ) -> None:
        cx, cy = self.transform.apply(x, y)
        radius = diameter / 2 * self.transform.scale
        if fill is not None:
            arcade.draw_circle_filled(cx, cy, radius, fill)
        if stroke is not None:
            arcade.draw_circle_outline(cx, cy, radius, stroke, weight * self.transform.scale)

    def ellipse(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: Color | None = WHITE,
        stroke: Color | None = BLACK,
        weight: float = 1,
    ) -> None:
        segments = 64
        points = [
            (x + width / 2 * math.cos(a), y + height / 2 * math.sin(a))
            for a in (2 * math.pi * i / segments for i in range(segments))
        ]
        self.polygon(points, fill, stroke, weight)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: Color | None = WHITE,
        stroke: Color | None = BLACK,
        weight: float = 1,
        *,
        center: bool = False,
    ) -> None:
        """A rectangle from its top-left corner, or from its center (p5's ``rectMode(CENTER)``)."""
        if center:
            x, y = x - width / 2, y - height / 2
        corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        self.polygon(corners, fill, stroke, weight)

    def polygon(
        self,
        points: Sequence[Point],
        fill: Color | None = WHITE,
        stroke: Color | None = BLACK,
        weight: float = 1,
    ) -> None:
        """A closed shape (p5's ``beginShape()`` ... ``endShape(CLOSE)``)."""
        mapped = [self.transform.apply(px, py) for px, py in points]
        if fill is not None:
            arcade.draw_polygon_filled(mapped, fill)
        if stroke is not None:
            arcade.draw_polygon_outline(mapped, stroke, weight * self.transform.scale)

    def polyline(self, points: Sequence[Point], stroke: Color = BLACK, weight: float = 1) -> None:
        """An open shape (p5's ``beginShape()`` ... ``endShape()`` with ``noFill()``)."""
        mapped = [self.transform.apply(px, py) for px, py in points]
        arcade.draw_line_strip(mapped, stroke, weight * self.transform.scale)

    def text(
        self,
        text: str,
        x: float,
        y: float,
        color: Color = BLACK,
        size: float = 12,
        *,
        align: str = "left",
        baseline: str = "baseline",
    ) -> None:
        """Text at ``(x, y)``; ``align`` and ``baseline`` follow arcade's ``anchor_x/anchor_y``."""
        px, py = self.transform.apply(x, y)
        key = (text, size, color, align, baseline)
        if (label := self._texts.get(key)) is None:
            if len(self._texts) > 256:
                self._texts.clear()
            label = arcade.Text(text, 0, 0, color, size, anchor_x=align, anchor_y=baseline)
            self._texts[key] = label
        label.position = (px, self.height - py)
        with self._text_camera.activate():
            label.draw()
        self._camera.use()


class Sketch(arcade.View):
    """One book example. Subclasses set it up in ``__init__`` (p5's ``setup()``)."""

    title: ClassVar[str] = "Sketch"
    help: ClassVar[Sequence[str]] = ()
    canvas_size: ClassVar[tuple[int, int]] = (640, 240)
    background: ClassVar[Color | None] = WHITE

    def __init__(self) -> None:
        super().__init__()
        self.paused = False
        self.show_help = False
        self.status = ""  # one-line readout drawn at the bottom
        self.frame_count = 0
        self.mouse: Point = (0.0, 0.0)  # book coordinates, like p5's mouseX/mouseY
        self.mouse_pressed = False
        self.held: set[int] = set()  # keys currently held down
        width, height = self.canvas_size
        self.canvas = Canvas(self.window, width, height, persistent=self.background is None)
        self._help_text = arcade.Text(
            "", 8, 0, arcade.color.WHITE, 11, width=round(width * SCALE) - 16,
            multiline=True, anchor_y="top",
        )  # fmt: skip
        self._status_text = arcade.Text("", 8, 6, arcade.color.BLACK, 11)

    # --- to override ----------------------------------------------------------------
    def step(self) -> None:
        """Advance the simulation by one frame."""

    def draw(self, canvas: Canvas) -> None:
        """Draw the current state (p5's ``draw()`` without the simulation part)."""

    # --- arcade callbacks -------------------------------------------------------------
    def on_fixed_update(self, delta_time: float) -> None:
        if not self.paused:
            self.step()
            self.frame_count += 1

    def on_draw(self) -> None:
        self.clear(color=self.background or WHITE)
        with self.canvas.frame():
            self.draw(self.canvas)
        self.window.default_camera.use()
        self._draw_overlay()

    def _draw_overlay(self) -> None:
        window = self.window
        if self.show_help:
            sketches = window.sketches if isinstance(window, SketchWindow) else ()
            lines = [self.title, *self.help, "P: pause  N: step  R: restart  H: help"]
            if len(sketches) > 1:
                lines.append("PageDown / PageUp: next / previous example")
            self._help_text.text = "\n".join(lines)
            self._help_text.y = window.height - 8
            box = self._help_text.content_height + 16
            arcade.draw_lrbt_rectangle_filled(
                0, window.width, window.height - box, window.height, (0, 0, 0, 170)
            )
            self._help_text.draw()
        status = ("[PAUSED]  " if self.paused else "") + self.status
        if status:
            self._status_text.text = status
            self._status_text.draw()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> bool | None:
        self.mouse = (x / SCALE, self.canvas.height - y / SCALE)
        return None

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> bool | None:
        return self.on_mouse_motion(x, y, dx, dy)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> bool | None:
        self.mouse_pressed = True
        return None

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> bool | None:
        self.mouse_pressed = False
        return None

    def on_key_release(self, symbol: int, modifiers: int) -> bool | None:
        self.held.discard(symbol)
        return None

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        self.held.add(symbol)
        window = self.window
        match symbol:
            case arcade.key.P:
                self.paused = not self.paused
            case arcade.key.N if self.paused:
                self.step()
                self.frame_count += 1
            case arcade.key.H:
                self.show_help = not self.show_help
            case arcade.key.R if isinstance(window, SketchWindow):
                window.show_sketch(window.index)
            case arcade.key.PAGEDOWN if isinstance(window, SketchWindow):
                window.show_sketch(window.index + 1)
            case arcade.key.PAGEUP if isinstance(window, SketchWindow):
                window.show_sketch(window.index - 1)
            case _:
                return False
        return True


class SketchWindow(arcade.Window):
    def __init__(self, sketches: Sequence[type[Sketch]], title: str) -> None:
        width, height = sketches[0].canvas_size
        super().__init__(round(width * SCALE), round(height * SCALE), title, fixed_rate=FIXED_RATE)
        self.sketches = sketches
        self.index = 0

    def show_sketch(self, index: int) -> None:
        self.index = index % len(self.sketches)
        cls = self.sketches[self.index]
        width, height = cls.canvas_size
        if (round(width * SCALE), round(height * SCALE)) != self.get_size():
            self.set_size(round(width * SCALE), round(height * SCALE))
        self.set_caption(f"{cls.title}  [{self.index + 1}/{len(self.sketches)}]")
        self.show_view(cls())


def run(*sketches: type[Sketch], title: str = "The Nature of Code") -> None:
    """Open a window with the given examples; ``python -m ... N`` starts at the N-th one."""
    window = SketchWindow(sketches, title)
    start = int(sys.argv[1]) - 1 if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    window.show_sketch(start)
    arcade.run()
