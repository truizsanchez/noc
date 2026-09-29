import math

import pytest

from noc.common.tessellate import (
    earclip,
    ellipse_points,
    fan,
    fill,
    is_convex,
    segments_for,
    stroke,
)

type Point = tuple[float, float]


def area(triangles: list[Point]) -> float:
    total = 0.0
    for i in range(0, len(triangles), 3):
        (ax, ay), (bx, by), (cx, cy) = triangles[i : i + 3]
        total += abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2
    return total


SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
# An L shape: concave at (5, 5).
ELL = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (5.0, 5.0), (5.0, 10.0), (0.0, 10.0)]


def test_fan_covers_a_convex_polygon() -> None:
    triangles = fan(SQUARE)
    assert len(triangles) == 6
    assert area(triangles) == pytest.approx(100)


def test_convexity() -> None:
    assert is_convex(SQUARE)
    assert is_convex(ellipse_points(0, 0, 5, 3, 32))
    assert not is_convex(ELL)


@pytest.mark.parametrize("points", [ELL, ELL[::-1]])
def test_earclip_covers_a_concave_polygon_in_either_winding(points: list[Point]) -> None:
    triangles = earclip(points)
    assert len(triangles) == 3 * (len(points) - 2)
    assert area(triangles) == pytest.approx(75)


def test_fill_picks_the_right_method() -> None:
    assert area(fill(SQUARE)) == pytest.approx(100)
    assert area(fill(ELL)) == pytest.approx(75)


def test_stroke_of_a_segment_is_a_rectangle_weight_wide() -> None:
    triangles = stroke([(0, 0), (10, 0)], 2)
    assert area(triangles) == pytest.approx(20)
    assert {round(y, 9) for _, y in triangles} == {-1, 1}


def test_closed_stroke_of_a_square_has_mitered_corners() -> None:
    # Outer square 12x12 minus inner 8x8.
    assert area(stroke(SQUARE, 2, closed=True)) == pytest.approx(144 - 64)


def test_open_polyline_joins_are_mitered_too() -> None:
    # An L-shaped path: two 10-long arms, 2 wide, sharing the corner square.
    triangles = stroke([(0, 0), (10, 0), (10, 10)], 2)
    assert area(triangles) == pytest.approx(20 + 20)


def test_repeated_points_are_ignored() -> None:
    assert stroke([(0, 0), (0, 0)], 1) == []
    assert area(stroke([(0, 0), (5, 0), (5, 0), (10, 0)], 2)) == pytest.approx(20)


def test_ellipse_points_and_segment_count() -> None:
    points = ellipse_points(1, 2, 3, 4, 4)
    assert points[0] == pytest.approx((4, 2))
    assert points[1] == pytest.approx((1, 6))
    assert segments_for(1) == 12
    assert segments_for(1000) == 96
    assert math.isclose(area(fan(ellipse_points(0, 0, 10, 10, 96))), math.pi * 100, rel_tol=0.01)
