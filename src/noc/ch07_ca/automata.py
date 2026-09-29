"""Cellular automata: Wolfram's elementary CA, the Game of Life, and variations.

Grids are numpy arrays indexed ``[row, column]`` (y first, like an image), so a whole
generation is computed with array operations instead of a loop over cells. Example 7.3's
object-oriented cells are kept too, for the book's point about cells with more state.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

type Grid = npt.NDArray[np.int8]
type FloatGrid = npt.NDArray[np.float64]


# --- elementary CA -------------------------------------------------------------------------
def ruleset(rule: int) -> list[int]:
    """Wolfram's rule number as the book's array: ``ruleset[7 - index]`` for neighborhood
    ``index`` (the three cells read as a binary number), so rule 90 is ``[0,1,0,1,1,0,1,0]``."""
    return [(rule >> (7 - i)) & 1 for i in range(8)]


def elementary_step(cells: Grid, rule: int, *, wrap: bool = False) -> Grid:
    """One generation. Without ``wrap`` the edge cells never change, as in Example 7.1."""
    left, right = np.roll(cells, 1), np.roll(cells, -1)
    index = (left << 2) | (cells << 1) | right
    table = np.array([(rule >> i) & 1 for i in range(8)], np.int8)  # table[index]
    nxt: Grid = table[index]
    if not wrap:
        nxt[0], nxt[-1] = cells[0], cells[-1]
    return nxt


def single_seed(width: int) -> Grid:
    """Generation 0 of Example 7.1: one live cell in the middle."""
    cells = np.zeros(width, np.int8)
    cells[width // 2] = 1
    return cells


def random_row(width: int, rng: random.Random) -> Grid:
    """Exercise 7.2: generation 0 at random."""
    return np.array([rng.randrange(2) for _ in range(width)], np.int8)


@dataclass
class History:
    """Exercise 7.4: the last ``length`` generations, oldest first, for a scrolling view."""

    cells: Grid
    rule: int
    length: int
    rows: list[Grid] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rows = [self.cells]

    def step(self) -> None:
        self.cells = elementary_step(self.cells, self.rule, wrap=True)
        self.rows = [*self.rows[-(self.length - 1) :], self.cells]

    def grid(self) -> Grid:
        return np.array(self.rows)


def continuous_step(cells: FloatGrid, rule: int, rate: float = 0.5) -> FloatGrid:
    """Exercise 7.10: float states. A cell counts as on when above 0.5; the rule gives its
    target (0 or 1) and the state moves ``rate`` of the way there."""
    binary = (cells > 0.5).astype(np.int8)
    target = elementary_step(binary, rule, wrap=True).astype(np.float64)
    return cells + (target - cells) * rate


# --- the Game of Life ----------------------------------------------------------------------
def neighbor_counts(board: Grid, *, wrap: bool = False) -> Grid:
    """Live neighbors of every cell (the 8 around it)."""
    if wrap:
        return sum(
            (np.roll(board, (dy, dx), axis=(0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if dy or dx),
            np.zeros_like(board),
        )  # fmt: skip
    padded = np.pad(board, 1)
    rows, cols = board.shape
    return sum(
        (padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
         for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx),
        np.zeros_like(board),
    )  # fmt: skip


def life_step(board: Grid, *, wrap: bool = False, out: Grid | None = None) -> Grid:
    """Example 7.2: birth with 3 neighbors, survival with 2 or 3.

    Without ``wrap`` the border cells stay dead, as in the book (Exercise 7.6 adds the
    wraparound). ``out`` receives the result instead of a new array (Exercise 7.7).
    """
    counts = neighbor_counts(board, wrap=wrap)
    result = out if out is not None else np.empty_like(board)
    result[...] = (counts == 3) | ((board == 1) & (counts == 2))
    if not wrap:
        result[0, :] = result[-1, :] = result[:, 0] = result[:, -1] = 0
    return result


@dataclass
class DoubleBuffer:
    """Exercise 7.7: two boards, swapped every generation instead of allocating a new one."""

    current: Grid
    wrap: bool = False
    spare: Grid = field(init=False)

    def __post_init__(self) -> None:
        self.spare = np.empty_like(self.current)

    def step(self) -> None:
        life_step(self.current, wrap=self.wrap, out=self.spare)
        self.current, self.spare = self.spare, self.current


def random_board(rows: int, cols: int, rng: random.Random) -> Grid:
    board = np.array([[rng.randrange(2) for _ in range(cols)] for _ in range(rows)], np.int8)
    board[0, :] = board[-1, :] = board[:, 0] = board[:, -1] = 0
    return board


PATTERNS: dict[str, list[str]] = {
    "glider": [".#.", "..#", "###"],
    "blinker": ["###"],
    "lightweight spaceship": [".#..#", "#....", "#...#", "####."],
    "pulsar": [
        "..###...###..",
        ".............",
        "#....#.#....#",
        "#....#.#....#",
        "#....#.#....#",
        "..###...###..",
        ".............",
        "..###...###..",
        "#....#.#....#",
        "#....#.#....#",
        "#....#.#....#",
        ".............",
        "..###...###..",
    ],
    "gosper glider gun": [
        "........................#...........",
        "......................#.#...........",
        "............##......##............##",
        "...........#...#....##............##",
        "##........#.....#...##..............",
        "##........#...#.##....#.#...........",
        "..........#.....#.......#...........",
        "...........#...#....................",
        "............##......................",
    ],
}


def stamp(board: Grid, pattern: str, row: int, col: int) -> None:
    """Exercise 7.5: draw a known pattern with its top-left corner at ``(row, col)``."""
    for dy, line in enumerate(PATTERNS[pattern]):
        for dx, char in enumerate(line):
            r, c = row + dy, col + dx
            if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                board[r, c] = char == "#"


def probabilistic_step(board: Grid, rng: np.random.Generator) -> Grid:
    """Exercise 7.9: the book's probabilistic rules.

    Overpopulation (4+ neighbors): 80% chance of dying. Loneliness (0-1): 60% chance of
    dying. The book leaves birth alone; here a dead cell with exactly 3 neighbors comes alive
    70% of the time ("make up your own"). Otherwise nothing changes.
    """
    counts = neighbor_counts(board, wrap=True)
    roll = rng.random(board.shape)
    alive = board == 1
    dies = alive & (((counts >= 4) & (roll < 0.8)) | ((counts <= 1) & (roll < 0.6)))
    born = ~alive & (counts == 3) & (roll >= 0.3)
    return np.where(dies, 0, np.where(born, 1, board)).astype(np.int8)


@dataclass
class Ages:
    """Exercise 7.12: how many generations each cell has been alive (positive) or dead
    (negative), for coloring by history."""

    board: Grid
    ages: npt.NDArray[np.int64] = field(init=False)

    def __post_init__(self) -> None:
        self.ages = np.where(self.board == 1, 1, -1).astype(np.int64)

    def step(self) -> None:
        nxt = life_step(self.board, wrap=True)
        same = nxt == self.board
        self.ages = np.where(same, self.ages + np.sign(self.ages), np.where(nxt == 1, 1, -1))
        self.board = nxt


# --- object-oriented cells ------------------------------------------------------------------
@dataclass
class Cell:
    """Example 7.3: a cell that remembers its previous state, for drawing changes."""

    state: int
    previous: int = 0

    def __post_init__(self) -> None:
        self.previous = self.state


@dataclass
class CellBoard:
    cells: list[list[Cell]]  # [row][column]

    @classmethod
    def random(cls, rows: int, cols: int, rng: random.Random) -> CellBoard:
        def state(r: int, c: int) -> int:
            border = r in (0, rows - 1) or c in (0, cols - 1)
            return 0 if border else rng.randrange(2)

        return cls([[Cell(state(r, c)) for c in range(cols)] for r in range(rows)])

    def advance(self) -> None:
        """One generation, computed from ``previous``; afterward each cell holds both its old
        (``previous``) and new (``state``) state, which is what the drawing compares."""
        for cell in self:
            cell.previous = cell.state
        cells = self.cells
        rows, cols = len(cells), len(cells[0])
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                neighbors = sum(
                    cells[r + dy][c + dx].previous
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if dy or dx
                )
                cell = cells[r][c]
                if cell.state == 1 and not 2 <= neighbors <= 3:
                    cell.state = 0
                elif cell.state == 0 and neighbors == 3:
                    cell.state = 1

    def states(self) -> Grid:
        return np.array([[cell.state for cell in row] for row in self.cells], np.int8)

    def __iter__(self) -> Iterator[Cell]:
        for row in self.cells:
            yield from row


# --- hexagons and pixels -------------------------------------------------------------------
def hex_neighbor_counts(board: Grid) -> Grid:
    """Exercise 7.8: six neighbors on an "odd-r" offset hexagon grid (odd rows shifted right),
    wrapping around the edges."""

    def shifted(dy: int, dx: int) -> Grid:
        return np.roll(board, (-dy, -dx), axis=(0, 1))

    even = shifted(0, -1) + shifted(0, 1) + shifted(-1, -1) + shifted(-1, 0)
    even = even + shifted(1, -1) + shifted(1, 0)
    odd = shifted(0, -1) + shifted(0, 1) + shifted(-1, 0) + shifted(-1, 1)
    odd = odd + shifted(1, 0) + shifted(1, 1)
    rows = np.arange(board.shape[0])[:, None]
    return np.where(rows % 2 == 0, even, odd).astype(np.int8)


def hex_life_step(board: Grid, birth: frozenset[int] = frozenset({2}),
                  survive: frozenset[int] = frozenset({3, 4})) -> Grid:  # fmt: skip
    """A Life-like rule for hexagons (B2/S34 by default, a well-behaved hexagonal rule)."""
    counts = hex_neighbor_counts(board)
    born = (board == 0) & np.isin(counts, list(birth))
    stays = (board == 1) & np.isin(counts, list(survive))
    result: Grid = (born | stays).astype(np.int8)
    return result


def cyclic_step(states: Grid, levels: int, threshold: int = 1) -> Grid:
    """Exercise 7.11: Griffeath's cyclic CA, one cell per pixel, the state as its color.

    A cell advances to the next state (mod ``levels``) when at least ``threshold`` of its 8
    neighbors are already in that next state, which grows spirals out of random noise.
    """
    successor = (states + 1) % levels
    count = np.zeros_like(states)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                count += np.roll(states, (dy, dx), axis=(0, 1)) == successor
    result: Grid = np.where(count >= threshold, successor, states).astype(np.int8)
    return result
