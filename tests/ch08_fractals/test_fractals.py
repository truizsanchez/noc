import math
import random

import pytest

from noc.ch08_fractals.fractals import (
    CantorLine,
    GrowingTree,
    KochLine,
    cantor,
    cantor_generations,
    circles_four_times,
    circles_twice,
    cross_lines,
    koch,
    nested_circles,
    sierpinski,
    snowflake,
    stochastic_tree,
    tree,
    windy_tree,
)
from noc.ch08_fractals.lsystem import PRESETS, LSystem, expand, turtle
from noc.common.noise import Noise
from noc.common.vector import Vector


def test_recursive_circles() -> None:
    radii = [r for _, _, r in nested_circles(0, 0, 100)]
    assert radii[0] == 100
    assert radii[1] == 75
    assert radii[-1] <= 4 < radii[-2]
    assert len(list(circles_twice(0, 0, 16))) == 1 + 2 + 4  # radii 16, 8, 4 (4 is not > 4)
    assert len(list(circles_four_times(0, 0, 64))) == 1 + 4 + 16  # radii 64, 32, 16


def test_cross_lines_alternate_direction() -> None:
    lines = list(cross_lines(Vector(0, 0), Vector(90, 0)))
    first, second = lines[0], lines[1]
    assert first == (Vector(0, 0), Vector(90, 0))
    assert second[0].x == second[1].x == 0  # a vertical line at the left end
    assert second[1].y - second[0].y == pytest.approx(60)


def test_cantor_thirds() -> None:
    lines = list(cantor(0, 0, 27))
    assert lines[:3] == [(0, 0, 27), (0, 20, 9), (0, 40, 3)]
    total = sum(length for _, y, length in lines if y == 40)
    assert total == pytest.approx(27 * 4 / 9)


def test_cantor_objects_match_the_recursion() -> None:
    generations = cantor_generations(CantorLine(10, 10, 620), 4)
    assert [len(g) for g in generations] == [1, 2, 4, 8]
    recursive = {(round(x, 6), y) for x, y, length in cantor(10, 10, 620) if y <= 70}
    objects = {(round(line.x, 6), line.y) for g in generations for line in g}
    assert recursive == objects


def test_koch_line_replacement() -> None:
    _, b, c, d, _ = KochLine(Vector(0, 0), Vector(3, 0)).koch_points()
    assert b.xy == pytest.approx((1, 0))
    assert d.xy == pytest.approx((2, 0))
    assert c.xy == pytest.approx((1.5, -math.sqrt(3) / 2))  # the bump points up
    lines = koch([KochLine(Vector(0, 0), Vector(640, 0))], 3)
    assert len(lines) == 4**3
    length = sum(line.start.dist(line.end) for line in lines)
    assert length == pytest.approx(640 * (4 / 3) ** 3)


def test_snowflake_is_closed() -> None:
    lines = snowflake(Vector(0, 0), 100, 2)
    assert len(lines) == 3 * 16
    assert lines[-1].end.xy == pytest.approx(lines[0].start.xy)


def test_sierpinski_counts() -> None:
    triangles = list(sierpinski(Vector(0, 0), Vector(4, 0), Vector(2, 3), 3))
    assert len(triangles) == 27


def test_tree_branches_split_and_shrink() -> None:
    branches = list(tree(Vector(0, 0), 80, math.pi / 6))
    assert branches[0].end.xy == pytest.approx((0, -80))
    lengths = sorted({round(b.length, 6) for b in branches}, reverse=True)
    assert lengths[1] == pytest.approx(80 * 0.67)
    # A full binary tree: 2^n - 1 branches for n levels.
    assert len(branches) == 2 ** len(lengths) - 1


def test_stochastic_tree_is_reproducible_with_a_seed() -> None:
    a = list(stochastic_tree(Vector(0, 0), 80, random.Random(1)))
    b = list(stochastic_tree(Vector(0, 0), 80, random.Random(1)))
    assert a == b
    assert a != list(stochastic_tree(Vector(0, 0), 80, random.Random(2)))


def test_windy_tree_keeps_its_shape_as_time_moves() -> None:
    noise = Noise(seed=3)
    a = list(windy_tree(Vector(0, 0), 60, noise, 0.0, random.Random(4)))
    b = list(windy_tree(Vector(0, 0), 60, noise, 0.1, random.Random(4)))
    assert len(a) == len(b)
    assert [x.length for x in a] == [x.length for x in b]
    assert a[-1].end != b[-1].end


def test_growing_tree_branches_then_grows_leaves() -> None:
    growing = GrowingTree.seed(Vector(320, 240))
    for _ in range(1000):
        growing.update()
    assert len(growing.branches) >= growing.max_branches
    assert growing.leaves


def test_lsystem_generations() -> None:
    system = LSystem("A", {"A": "AB", "B": "A"})
    sentences = []
    for _ in range(5):
        sentences.append(system.sentence)
        system.generate()
    assert sentences == ["A", "AB", "ABA", "ABAAB", "ABAABABA"]


def test_stochastic_rules_pick_among_options() -> None:
    system = LSystem("FFFFFFFFFF", {"F": [(1, "X"), (1, "Y")]}, random.Random(5))
    system.generate()
    assert set(system.sentence) == {"X", "Y"}


def test_turtle_with_brackets() -> None:
    segments = list(turtle("F[+F]-F", 10, math.pi / 2, Vector(0, 0)))
    assert len(segments) == 3
    (_, b1, _), (_, b2, d2), (_, b3, _) = segments
    assert b1.xy == pytest.approx((0, -10))
    assert b2.xy == pytest.approx((10, -10))  # turned clockwise on screen, inside a branch
    assert d2 == 1
    assert b3.xy == pytest.approx((-10, -10))  # back at the branch point, turned the other way
    start, _, _ = next(turtle("GF", 5, 0.0, Vector(0, 0)))
    assert start.xy == pytest.approx((0, -5))


@pytest.mark.parametrize("preset", PRESETS, ids=[p.name for p in PRESETS])
def test_presets_expand(preset: object) -> None:
    sentence = expand(preset, random.Random(6))  # type: ignore[arg-type]
    assert "F" in sentence
