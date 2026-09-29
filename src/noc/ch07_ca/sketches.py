"""Chapter 7's examples and exercises, in book order."""

from __future__ import annotations

import math
import random

import arcade
import numpy as np
import numpy.typing as npt

from noc.ch07_ca.automata import (
    PATTERNS,
    Ages,
    CellBoard,
    DoubleBuffer,
    Grid,
    History,
    continuous_step,
    cyclic_step,
    elementary_step,
    hex_life_step,
    probabilistic_step,
    random_board,
    random_row,
    single_seed,
    stamp,
)
from noc.common.view import Canvas, Sketch, gray

type Image = npt.NDArray[np.uint8]


def cells_image(colors: Image, size: int) -> Image:
    """Blow up one color per cell to ``size x size`` pixels."""
    return np.repeat(np.repeat(colors, size, axis=0), size, axis=1)


def draw_grid_lines(canvas: Canvas, size: int, rows: int, cols: int) -> None:
    for c in range(cols + 1):
        canvas.line(c * size, 0, c * size, rows * size, gray(0, 90))
    for r in range(rows + 1):
        canvas.line(0, r * size, cols * size, r * size, gray(0, 90))


def binary_colors(board: Grid) -> Image:
    """Live cells black, dead cells white."""
    value = (255 - board.astype(np.uint8) * 255)[..., None]
    return np.repeat(value, 3, axis=2)


# --- elementary CA -------------------------------------------------------------------------
class ElementaryCA(Sketch):
    title = "Example 7.1: Wolfram Elementary Cellular Automata"
    help = ("R: new random rule (Exercise 7.1)   S: random first generation (Exercise 7.2)",)
    background = None
    cell_size = 10

    def __init__(self) -> None:
        super().__init__()
        self.rule = 90
        self.random_start = False
        self.restart()

    def restart(self) -> None:
        width = self.canvas.width // self.cell_size
        rng = random.Random()
        self.cells = random_row(width, rng) if self.random_start else single_seed(width)
        self.generation = 0
        self.needs_clear = True
        self.pending: list[tuple[int, Grid]] = []  # rows computed but not drawn yet

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.rule = random.randrange(256)
        elif symbol == arcade.key.S:
            self.random_start = not self.random_start
        else:
            return super().on_key_press(symbol, modifiers)
        self.restart()
        return True

    def step(self) -> None:
        if self.generation * self.cell_size > self.canvas.height:
            # Exercise 7.1: once the canvas is full, start over with a random rule.
            self.rule = random.randrange(256)
            self.restart()
        self.pending.append((self.generation, self.cells))
        self.cells = elementary_step(self.cells, self.rule)
        self.generation += 1

    def draw(self, canvas: Canvas) -> None:
        if self.needs_clear:
            canvas.background(gray(255))
            self.needs_clear = False
        for generation, cells in self.pending:
            for i, alive in enumerate(cells.tolist()):
                if alive:
                    y = generation * self.cell_size
                    canvas.rect(i * self.cell_size, y, self.cell_size, self.cell_size, fill=gray(0))
        self.pending = []
        self.status = f"rule {self.rule}"


class ScrollingCA(Sketch):
    title = "Exercise 7.4: An Elementary CA Scrolling to Infinity"
    help = ("R: new random rule   Up/Down: next/previous rule",)
    cell_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.start(30)

    def start(self, rule: int) -> None:
        width, height = self.canvas.width // self.cell_size, self.canvas.height // self.cell_size
        self.history = History(single_seed(width), rule % 256, height)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        match symbol:
            case arcade.key.R:
                self.start(random.randrange(256))
            case arcade.key.UP | arcade.key.DOWN:
                self.start(self.history.rule + (1 if symbol == arcade.key.UP else -1))
            case _:
                return super().on_key_press(symbol, modifiers)
        return True

    def step(self) -> None:
        self.history.step()

    def draw(self, canvas: Canvas) -> None:
        grid = self.history.grid()
        rows = canvas.height // self.cell_size
        padded = np.zeros((rows, grid.shape[1]), np.int8)
        padded[-len(grid) :] = grid
        canvas.pixels(cells_image(binary_colors(padded), self.cell_size), smooth=False)
        self.status = f"rule {self.history.rule}"


class ContinuousCA(Sketch):
    title = "Exercise 7.10: An Elementary CA with Float States"
    help = ("R: new random rule",)
    cell_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.rule = 30
        self.reset()

    def reset(self) -> None:
        width = self.canvas.width // self.cell_size
        self.cells = np.array([random.random() for _ in range(width)])
        self.rows: list[npt.NDArray[np.float64]] = [self.cells]

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.rule = random.randrange(256)
            self.reset()
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        self.cells = continuous_step(self.cells, self.rule, 0.4)
        self.rows = [*self.rows[-(self.canvas.height // self.cell_size - 1) :], self.cells]

    def draw(self, canvas: Canvas) -> None:
        rows = canvas.height // self.cell_size
        grid = np.ones((rows, len(self.cells)))
        grid[-len(self.rows) :] = np.array(self.rows)
        gray_levels = (255 - grid * 255).astype(np.uint8)[..., None]
        canvas.pixels(cells_image(np.repeat(gray_levels, 3, axis=2), self.cell_size), smooth=False)
        self.status = f"rule {self.rule}"


# --- the Game of Life ----------------------------------------------------------------------
class GameOfLife(Sketch):
    title = "Example 7.2: Game of Life"
    help = (
        "W: wrap around the edges (Exercise 7.6)   Space: pause   R: random board",
        "Click: toggle a cell   1-5: stamp a pattern at the mouse (Exercise 7.5)",
    )
    cell_size = 8

    def __init__(self) -> None:
        super().__init__()
        self.rows, self.cols = (
            self.canvas.height // self.cell_size,
            self.canvas.width // self.cell_size,
        )
        self.board = DoubleBuffer(random_board(self.rows, self.cols, random.Random()))
        self.running = True

    def cell_at_mouse(self) -> tuple[int, int]:
        x, y = self.mouse
        return int(y // self.cell_size), int(x // self.cell_size)

    def mouse_down(self) -> None:
        r, c = self.cell_at_mouse()
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.board.current[r, c] ^= 1

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        patterns = list(PATTERNS)
        if symbol == arcade.key.W:
            self.board.wrap = not self.board.wrap
        elif symbol == arcade.key.SPACE:
            self.running = not self.running
        elif symbol == arcade.key.R:
            self.board.current[...] = random_board(self.rows, self.cols, random.Random())
        elif arcade.key.KEY_1 <= symbol < arcade.key.KEY_1 + len(patterns):
            stamp(self.board.current, patterns[symbol - arcade.key.KEY_1], *self.cell_at_mouse())
        else:
            return super().on_key_press(symbol, modifiers)
        return True

    def step(self) -> None:
        if self.running:
            self.board.step()

    def draw(self, canvas: Canvas) -> None:
        canvas.pixels(cells_image(binary_colors(self.board.current), self.cell_size), smooth=False)
        draw_grid_lines(canvas, self.cell_size, self.rows, self.cols)
        state = "wrapping" if self.board.wrap else "dead edges"
        self.status = f"{state}{'' if self.running else '  [stopped]'}  patterns: " + ", ".join(
            f"{i + 1} {name}" for i, name in enumerate(PATTERNS)
        )


class ObjectOrientedLife(Sketch):
    title = "Example 7.3: Object-Oriented Game of Life"
    help = ("Blue: just born   Red: just died",)
    cell_size = 8

    def __init__(self) -> None:
        super().__init__()
        rows, cols = self.canvas.height // self.cell_size, self.canvas.width // self.cell_size
        self.board = CellBoard.random(rows, cols, random.Random())

    def step(self) -> None:
        self.board.advance()

    def draw(self, canvas: Canvas) -> None:
        colors = np.array(
            [
                [
                    (0, 0, 255) if (c.previous, c.state) == (0, 1)
                    else (0, 0, 0) if c.state == 1
                    else (255, 0, 0) if c.previous == 1
                    else (255, 255, 255)
                    for c in row
                ]
                for row in self.board.cells
            ],
            np.uint8,
        )  # fmt: skip
        canvas.pixels(cells_image(colors, self.cell_size), smooth=False)
        draw_grid_lines(canvas, self.cell_size, len(self.board.cells), len(self.board.cells[0]))


class HexagonLife(Sketch):
    title = "Exercise 7.8: A Hexagonal CA (B2/S34)"
    help = ("R: random board",)
    radius = 8.0

    def __init__(self) -> None:
        super().__init__()
        self.cols = int(self.canvas.width / (math.sqrt(3) * self.radius)) + 1
        rows = int(self.canvas.height / (1.5 * self.radius)) + 1
        self.rows = rows + rows % 2  # an even count, so the offset rows wrap around cleanly
        self.reset()

    def reset(self) -> None:
        rng = np.random.default_rng()
        self.board = (rng.random((self.rows, self.cols)) < 0.3).astype(np.int8)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.reset()
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        if self.frame_count % 6 == 0:
            self.board = hex_life_step(self.board)

    def draw(self, canvas: Canvas) -> None:
        r = self.radius
        width = math.sqrt(3) * r
        corners = [
            (r * math.cos(a), r * math.sin(a))
            for a in (math.pi / 6 + k * math.pi / 3 for k in range(6))
        ]
        canvas.background(gray(220))
        for row in range(self.rows):
            for col in range(self.cols):
                x = col * width + (width / 2 if row % 2 else 0)
                y = row * 1.5 * r
                fill = gray(0) if self.board[row, col] else gray(255)
                canvas.polygon(
                    [(x + dx, y + dy) for dx, dy in corners], fill=fill, stroke=gray(150)
                )


class ProbabilisticLife(Sketch):
    title = "Exercise 7.9: A Probabilistic Game of Life"
    cell_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng()
        rows, cols = self.canvas.height // self.cell_size, self.canvas.width // self.cell_size
        self.board = (self.rng.random((rows, cols)) < 0.5).astype(np.int8)

    def step(self) -> None:
        self.board = probabilistic_step(self.board, self.rng)

    def draw(self, canvas: Canvas) -> None:
        canvas.pixels(cells_image(binary_colors(self.board), self.cell_size), smooth=False)


class CyclicCA(Sketch):
    title = "Exercise 7.11: One Cell per Pixel (a Cyclic CA)"
    help = ("R: new random start",)
    canvas_size = (320, 120)
    levels = 14

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        hues = np.linspace(0, 1, self.levels, endpoint=False)
        self.palette = np.stack(
            [(np.sin(2 * np.pi * (hues + k / 3)) * 0.5 + 0.5) * 255 for k in range(3)], axis=1
        ).astype(np.uint8)

    def reset(self) -> None:
        rng = np.random.default_rng()
        width, height = self.canvas_size
        self.states = rng.integers(0, self.levels, (height, width)).astype(np.int8)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.reset()
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        self.states = cyclic_step(self.states, self.levels)

    def draw(self, canvas: Canvas) -> None:
        canvas.pixels(self.palette[self.states], smooth=False)


class HistoricalLife(Sketch):
    title = "Exercise 7.12: Cells Colored by How Long They've Been Alive or Dead"
    help = ("Bright: long-lived   Dark: long dead   R: random board",)
    cell_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        rows, cols = self.canvas.height // self.cell_size, self.canvas.width // self.cell_size
        self.ages = Ages(random_board(rows, cols, random.Random()))

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.R:
            self.reset()
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        self.ages.step()

    def draw(self, canvas: Canvas) -> None:
        ages = self.ages.ages
        alive = np.clip(ages, 0, 60) / 60
        dead = np.clip(-ages, 0, 60) / 60
        colors = np.zeros((*ages.shape, 3))
        colors[..., 0] = np.where(ages > 0, 255 * (0.3 + 0.7 * alive), 30 * (1 - dead))
        colors[..., 1] = np.where(ages > 0, 200 * alive, 20 * (1 - dead))
        colors[..., 2] = np.where(ages > 0, 80, 60 * (1 - dead))
        canvas.pixels(cells_image(colors.astype(np.uint8), self.cell_size), smooth=False)


SKETCHES: tuple[type[Sketch], ...] = (
    ElementaryCA,
    ScrollingCA,
    ContinuousCA,
    GameOfLife,
    ObjectOrientedLife,
    HexagonLife,
    ProbabilisticLife,
    CyclicCA,
    HistoricalLife,
)
