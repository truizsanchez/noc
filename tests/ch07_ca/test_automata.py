import random

import numpy as np
import pytest

from noc.ch07_ca.automata import (
    PATTERNS,
    Ages,
    CellBoard,
    DoubleBuffer,
    History,
    continuous_step,
    cyclic_step,
    elementary_step,
    hex_life_step,
    hex_neighbor_counts,
    life_step,
    neighbor_counts,
    probabilistic_step,
    random_board,
    ruleset,
    single_seed,
    stamp,
)


def test_ruleset_matches_the_books_array() -> None:
    assert ruleset(90) == [0, 1, 0, 1, 1, 0, 1, 0]
    assert ruleset(30) == [0, 0, 0, 1, 1, 1, 1, 0]


def test_rule_90_draws_sierpinski() -> None:
    cells = single_seed(9)
    cells = elementary_step(cells, 90)
    assert cells.tolist() == [0, 0, 0, 1, 0, 1, 0, 0, 0]
    cells = elementary_step(cells, 90)
    assert cells.tolist() == [0, 0, 1, 0, 0, 0, 1, 0, 0]


def test_rule_30_neighborhoods() -> None:
    table = ruleset(30)
    for index in range(8):
        left, me, right = (index >> 2) & 1, (index >> 1) & 1, index & 1
        cells = np.array([0, left, me, right, 0], np.int8)
        assert elementary_step(cells, 30)[2] == table[7 - index]


def test_edges_stay_fixed_unless_wrapping() -> None:
    cells = np.array([1, 0, 0, 0, 1], np.int8)
    assert elementary_step(cells, 255)[[0, -1]].tolist() == [1, 1]
    assert elementary_step(np.zeros(5, np.int8), 1, wrap=True).tolist() == [1] * 5


def test_history_keeps_the_last_generations() -> None:
    history = History(single_seed(11), 90, 4)
    for _ in range(10):
        history.step()
    grid = history.grid()
    assert grid.shape == (4, 11)
    assert (grid[-1] == history.cells).all()


def test_continuous_states_move_toward_the_rule() -> None:
    cells = np.array([0.0, 0.9, 0.2, 0.7])
    nxt = continuous_step(cells, 90, 0.5)
    assert ((nxt >= 0) & (nxt <= 1)).all()
    binary = (cells > 0.5).astype(np.int8)
    target = elementary_step(binary, 90, wrap=True)
    assert np.allclose(nxt, cells + (target - cells) * 0.5)


def test_blinker_oscillates() -> None:
    board = np.zeros((5, 5), np.int8)
    board[2, 1:4] = 1
    once = life_step(board)
    assert once[1:4, 2].tolist() == [1, 1, 1]
    assert once.sum() == 3
    assert (life_step(once) == board).all()


def test_neighbor_counts() -> None:
    board = np.zeros((4, 4), np.int8)
    board[0, 0] = 1
    assert neighbor_counts(board)[1, 1] == 1
    assert neighbor_counts(board)[3, 3] == 0
    assert neighbor_counts(board, wrap=True)[3, 3] == 1


def test_glider_travels_with_wraparound() -> None:
    board = np.zeros((8, 8), np.int8)
    stamp(board, "glider", 0, 0)
    start = board.copy()
    for _ in range(4 * 8):  # one diagonal cell every 4 generations
        board = life_step(board, wrap=True)
    assert (board == start).all()


def test_double_buffer_matches_new_arrays_and_reuses_two() -> None:
    rng = random.Random(1)
    board = random_board(20, 30, rng)
    buffer = DoubleBuffer(board.copy())
    arrays = {id(buffer.current), id(buffer.spare)}
    for _ in range(10):
        board = life_step(board)
        buffer.step()
        assert (buffer.current == board).all()
    assert {id(buffer.current), id(buffer.spare)} == arrays


def test_random_boards_have_dead_borders() -> None:
    board = random_board(10, 10, random.Random(2))
    assert board[0].sum() == board[-1].sum() == board[:, 0].sum() == board[:, -1].sum() == 0


def test_patterns_are_stamped() -> None:
    board = np.zeros((20, 50), np.int8)
    stamp(board, "gosper glider gun", 1, 1)
    assert board.sum() == sum(line.count("#") for line in PATTERNS["gosper glider gun"])


def test_object_cells_agree_with_the_array_version() -> None:
    cells = CellBoard.random(12, 16, random.Random(3))
    board = cells.states()
    for _ in range(5):
        cells.advance()
        board = life_step(board)
        assert (cells.states() == board).all()
    changed = [c for c in cells if c.previous != c.state]
    assert changed  # something was born or died


def test_probabilistic_life_is_random_but_bounded() -> None:
    rng = np.random.default_rng(4)
    board = (rng.random((30, 30)) < 0.5).astype(np.int8)
    a = probabilistic_step(board, np.random.default_rng(5))
    b = probabilistic_step(board, np.random.default_rng(6))
    assert not (a == b).all()
    assert set(np.unique(a).tolist()) <= {0, 1}


def test_ages_count_generations_alive_and_dead() -> None:
    board = np.zeros((5, 5), np.int8)
    board[1:3, 1:3] = 1  # a block: still life
    ages = Ages(board)
    for _ in range(3):
        ages.step()
    assert ages.ages[1, 1] == 4
    assert ages.ages[0, 0] == -4


def test_hexagon_neighbors() -> None:
    board = np.zeros((6, 6), np.int8)
    board[2, 2] = 1
    counts = hex_neighbor_counts(board)
    assert counts.sum() == 6
    assert counts[2, 1] == counts[2, 3] == 1  # same row
    assert counts[1, 1] == counts[1, 2] == 1  # even row 2: up-left and up
    assert counts[1, 3] == 0


def test_hex_life_rule() -> None:
    board = np.zeros((6, 6), np.int8)
    board[2, 2] = board[2, 3] = 1
    nxt = hex_life_step(board)
    # Cells touching both live cells are born (B2); the pair itself dies (1 neighbor each).
    assert nxt[2, 2] == 0
    assert nxt[1, 2] == 1


@pytest.mark.parametrize("levels", [4, 14])
def test_cyclic_ca_only_advances_by_one(levels: int) -> None:
    rng = np.random.default_rng(7)
    states = rng.integers(0, levels, (20, 20)).astype(np.int8)
    nxt = cyclic_step(states, levels)
    moved = nxt != states
    assert moved.any()
    assert ((nxt[moved] - states[moved]) % levels == 1).all()
