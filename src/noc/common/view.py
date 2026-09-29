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

import functools
import itertools
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import ClassVar

import arcade
import numpy as np
import numpy.typing as npt
from arcade.gl import BufferDescription, Texture2D, geometry
from arcade.types import LRBT
from PIL import Image

from noc.common import tessellate
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


def rgba(color: Color) -> tuple[int, int, int, int]:
    match color:
        case (r, g, b):
            return (r, g, b, 255)
        case (r, g, b, a):
            return (r, g, b, a)


class _Batch:
    """Colored triangles collected during a frame and drawn with one call.

    Positions go into a flat list and colors into runs of ``(color, vertex count)``, expanded
    with numpy when the batch is drawn: much cheaper than building each vertex in Python.
    """

    def __init__(self, ctx: arcade.ArcadeContext) -> None:
        self._ctx = ctx
        self._program = ctx.shape_element_list_program
        self._positions: list[float] = []
        self._colors: list[tuple[int, int, int, int]] = []
        self._counts: list[int] = []
        self._vbo = ctx.buffer(reserve=4096 * 8, usage="stream")
        self._cbo = ctx.buffer(reserve=4096 * 16, usage="stream")
        self._geometry = ctx.geometry(
            [
                BufferDescription(self._vbo, "2f", ["in_vert"]),
                BufferDescription(self._cbo, "4f", ["in_color"]),
            ]
        )

    def __bool__(self) -> bool:
        return bool(self._positions)

    def add(self, triangles: Sequence[Point], color: Color) -> None:
        self._positions.extend(itertools.chain.from_iterable(triangles))
        self._colors.append(rgba(color))
        self._counts.append(len(triangles))

    def flush(self, blend: tuple[int, int]) -> None:
        if not self._positions:
            return
        positions = np.array(self._positions, np.float32)
        # Colors stay 0-255, as arcade's shape shader expects.
        colors = np.repeat(np.array(self._colors, np.float32), self._counts, axis=0)
        for buffer, data in ((self._vbo, positions), (self._cbo, colors)):
            if data.nbytes > buffer.size:
                buffer.orphan(size=data.nbytes * 2)
            buffer.write(data.tobytes())
        self._program["Position"] = 0.0, 0.0
        self._program["Angle"] = 0.0
        with self._ctx.enabled(self._ctx.BLEND):
            self._ctx.blend_func = blend
            self._geometry.render(self._program, vertices=len(positions) // 2)
            self._ctx.blend_func = self._ctx.BLEND_DEFAULT
        self._positions, self._colors, self._counts = [], [], []


class _ImageBatch:
    """Textured quads (p5's ``image()``) collected during a frame, drawn as one sprite list."""

    def __init__(self) -> None:
        self._sprites: arcade.SpriteList[arcade.Sprite] = arcade.SpriteList()
        self._pending: list[tuple[arcade.Texture, float, float, float, float, Color]] = []

    def __bool__(self) -> bool:
        return bool(self._pending)

    def add(
        self, texture: arcade.Texture, x: float, y: float, w: float, h: float, tint: Color
    ) -> None:
        self._pending.append((texture, x, y, w, h, tint))

    def flush(self, blend: tuple[int, int]) -> None:
        if not self._pending:
            return
        sprites = self._sprites
        while len(sprites) < len(self._pending):
            sprites.append(arcade.Sprite(self._pending[0][0]))
        while len(sprites) > len(self._pending):
            sprites.pop()
        for sprite, (texture, x, y, w, h, tint) in zip(sprites, self._pending, strict=True):
            if sprite.texture is not texture:
                sprite.texture = texture
            sprite.position = (x, y)
            sprite.width, sprite.height = w, h
            sprite.color = rgba(tint)
        sprites.draw(blend_function=blend)
        self._pending = []


def make_texture(image: Image.Image, name: str) -> arcade.Texture:
    """A texture for :meth:`Canvas.image`, from a PIL image with rows from the top."""
    # The canvas camera points y down, which would draw the image upside down.
    flipped = image.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return arcade.Texture(flipped, hash=f"noc-{name}")


@functools.cache
def disc_texture(ring: float) -> arcade.Texture:
    """A white disc with a black outline ``ring`` of its radius wide, for :meth:`Canvas.discs`."""
    size = 64
    coords = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(coords, coords)
    r = np.hypot(x, y)
    edge = 2 / size  # one texel of antialiasing
    alpha = np.clip((1 - r) / edge, 0, 1)
    white = np.clip((1 - ring - r) / edge, 0, 1)
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = (white * 255)[..., None].astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    return arcade.Texture(Image.fromarray(rgba), hash=f"noc-disc-{ring:.3f}")


class Canvas:
    """p5-style drawing in book coordinates, with explicit colors instead of global state.

    ``fill=None`` is ``noFill()`` and ``stroke=None`` is ``noStroke()``; the defaults are
    p5's (white fill, black 1-pixel stroke). Shapes are turned into triangles
    (:mod:`noc.common.tessellate`) and drawn together at the end of the frame, or before
    anything that isn't a shape (text, images).
    """

    def __init__(self, window: arcade.Window, width: int, height: int, *, persistent: bool) -> None:
        self.width, self.height = width, height
        self.transform = IDENTITY
        pixels = (round(width * SCALE), round(height * SCALE))
        self._pixel_scale = pixels[0] / width
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
        self._batch = _Batch(self._ctx)
        self._image_batch = _ImageBatch()
        self.additive = False  # p5's blendMode(ADD); reset every frame
        self._quad = geometry.quad_2d_fs()
        self._images: dict[tuple[int, int], Texture2D] = {}
        self._fbo = None
        if persistent:
            self._texture = self._ctx.texture(pixels, components=4)
            self._fbo = self._ctx.framebuffer(color_attachments=[self._texture])
            self._fbo.clear(color=(*WHITE, 255))

    @contextmanager
    def frame(self) -> Iterator[None]:
        """Draw one frame; on a persistent canvas, into the texture that keeps the old ones."""
        self.transform = IDENTITY
        self.additive = False
        if self._fbo is None:
            with self._camera.activate():
                yield
                self._flush()
            return
        with self._fbo.activate(), self._camera.activate():
            yield
            self._flush()
        self._blit(self._texture)

    def _flush(self) -> None:
        # Additive blending weighted by alpha (p5's ADD): arcade's BLEND_ADDITIVE is (ONE, ONE),
        # which would add a texture's full color even where it's transparent.
        ctx = self._ctx
        blend = (ctx.SRC_ALPHA, ctx.ONE) if self.additive else ctx.BLEND_DEFAULT
        self._image_batch.flush(blend)
        self._batch.flush(blend)

    def set_additive(self, additive: bool) -> None:
        """Switch additive blending on or off for what's drawn next (p5's ``blendMode()``)."""
        if additive != self.additive:
            self._flush()
            self.additive = additive

    def _blit(self, texture: Texture2D) -> None:
        # Copy without blending: a persistent canvas's alpha is below 1 wherever translucent
        # shapes were drawn, and blending it over the window would wash those colors out.
        texture.use(0)
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
    def _map(self, points: Sequence[Point]) -> list[Point]:
        if self.transform.is_identity:
            return list(points)
        apply = self.transform.apply
        return [apply(x, y) for x, y in points]

    def _before_shapes(self) -> None:
        if self._image_batch:
            self._flush()

    def _shape(
        self,
        points: list[Point],
        fill: Color | None,
        stroke: Color | None,
        weight: float,
        *,
        convex: bool,
    ) -> None:
        self._before_shapes()
        if fill is not None:
            self._batch.add(tessellate.fan(points) if convex else tessellate.fill(points), fill)
        if stroke is not None:
            self._batch.add(
                tessellate.stroke(points, weight * self.transform.scale, closed=True), stroke
            )

    def background(self, color: Color) -> None:
        self._before_shapes()
        corners = [(0.0, 0.0), (self.width, 0.0), (self.width, self.height), (0.0, self.height)]
        self._batch.add(tessellate.fan(corners), color)

    def line(
        self, x1: float, y1: float, x2: float, y2: float, stroke: Color = BLACK, weight: float = 1
    ) -> None:
        self.polyline([(x1, y1), (x2, y2)], stroke, weight)

    def point(self, x: float, y: float, stroke: Color = BLACK, weight: float = 1) -> None:
        """A dot ``weight`` pixels across (p5 draws points with the stroke)."""
        self._before_shapes()
        size = weight * self.transform.scale
        px, py = self.transform.apply(x, y)
        if size * self._pixel_scale <= 3:
            h = size / 2
            square = [(px - h, py - h), (px + h, py - h), (px + h, py + h), (px - h, py + h)]
            self._batch.add(tessellate.fan(square), stroke)
        else:
            ring = tessellate.ellipse_points(px, py, size / 2, size / 2, 12)
            self._batch.add(tessellate.fan(ring), stroke)

    def circle(
        self,
        x: float,
        y: float,
        diameter: float,
        fill: Color | None = WHITE,
        stroke: Color | None = BLACK,
        weight: float = 1,
    ) -> None:
        self.ellipse(x, y, diameter, diameter, fill, stroke, weight)

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
        scale = self.transform.scale
        segments = tessellate.segments_for(max(width, height) / 2 * scale * self._pixel_scale)
        if width != height and self.transform.angle:
            outline = tessellate.ellipse_points(x, y, width / 2, height / 2, segments)
            self._shape(self._map(outline), fill, stroke, weight, convex=True)
            return
        # Circles (or unrotated ellipses) keep their shape: transform only the center.
        self._before_shapes()
        cx, cy = self.transform.apply(x, y)
        rx, ry = width / 2 * scale, height / 2 * scale
        if fill is not None:
            outline = tessellate.ellipse_points(cx, cy, rx, ry, segments)
            self._batch.add(tessellate.fan(outline), fill)
        if stroke is not None:
            self._batch.add(tessellate.ring(cx, cy, rx, ry, weight * scale, segments), stroke)

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
        self._shape(self._map(corners), fill, stroke, weight, convex=True)

    def polygon(
        self,
        points: Sequence[Point],
        fill: Color | None = WHITE,
        stroke: Color | None = BLACK,
        weight: float = 1,
    ) -> None:
        """A closed shape (p5's ``beginShape()`` ... ``endShape(CLOSE)``), convex or not."""
        if len(points) >= 3:
            self._shape(self._map(points), fill, stroke, weight, convex=False)

    def polyline(self, points: Sequence[Point], stroke: Color = BLACK, weight: float = 1) -> None:
        """An open shape (p5's ``beginShape()`` ... ``endShape()`` with ``noFill()``)."""
        self._before_shapes()
        self._batch.add(tessellate.stroke(self._map(points), weight * self.transform.scale), stroke)

    def discs(
        self, centers: Sequence[Point], diameter: float, fills: Sequence[Color], weight: float = 1
    ) -> None:
        """Many same-size circles with a black outline, drawn as sprites.

        Equivalent to calling :meth:`circle` for each center with its fill, but a thousand
        of them cost about as much as a few tessellated circles.
        """
        ring = round(min(weight * 2 / diameter, 1.0), 3)
        texture = disc_texture(ring)
        for (x, y), fill in zip(centers, fills, strict=True):
            self.image(texture, x, y, diameter + weight, tint=fill)

    # --- things that aren't shapes -----------------------------------------------------
    def image(
        self,
        texture: arcade.Texture,
        x: float,
        y: float,
        width: float,
        height: float | None = None,
        tint: Color = WHITE,
    ) -> None:
        """A texture centered at ``(x, y)`` (p5's ``imageMode(CENTER)``), tinted like ``tint()``."""
        if self._batch:
            self._flush()
        scale = self.transform.scale
        px, py = self.transform.apply(x, y)
        h = width if height is None else height
        self._image_batch.add(texture, px, py, width * scale, h * scale, tint)

    def pixels(self, image: npt.NDArray[np.uint8], *, smooth: bool = True) -> None:
        """Cover the canvas with an RGB or RGBA image, rows from the top (p5's ``updatePixels()``).

        The image may have any resolution; it is stretched to the canvas, blending between
        pixels, or with ``smooth=False`` as crisp blocks (grids of cells).
        """
        self._flush()
        height, width = image.shape[:2]
        if image.shape[2] == 3:
            alpha = np.full((height, width, 1), 255, np.uint8)
            image = np.concatenate([image, alpha], axis=2)
        if (texture := self._images.get((width, height))) is None:
            texture = self._ctx.texture((width, height), components=4)
            self._images[width, height] = texture
        mode = self._ctx.LINEAR if smooth else self._ctx.NEAREST
        texture.filter = (mode, mode)
        texture.write(np.ascontiguousarray(image[::-1]).tobytes())
        self._blit(texture)

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
        self._flush()
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

    def mouse_down(self) -> None:
        """A mouse button was pressed at ``self.mouse`` (p5's ``mousePressed()``)."""

    def mouse_up(self) -> None:
        """The mouse button was released (p5's ``mouseReleased()``)."""

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
        self.on_mouse_motion(x, y, 0, 0)
        self.mouse_pressed = True
        self.mouse_down()
        return None

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> bool | None:
        self.mouse_pressed = False
        self.mouse_up()
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
