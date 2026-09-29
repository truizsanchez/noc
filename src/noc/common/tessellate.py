"""Shapes to triangles, so a whole frame can go to the GPU in one draw call.

Drawing each shape with its own arcade call costs ~30 microseconds, which adds up to more
than a frame's budget with a few hundred shapes. :class:`~noc.common.view.Canvas` instead
turns every fill and stroke into triangles with these functions and draws them together.
All functions return a flat list of vertices, three per triangle.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

type Point = tuple[float, float]

MITER_LIMIT = 4.0  # as a multiple of half the stroke weight, like SVG's default


def ellipse_points(cx: float, cy: float, rx: float, ry: float, segments: int) -> list[Point]:
    step = 2 * math.pi / segments
    return [(cx + rx * math.cos(i * step), cy + ry * math.sin(i * step)) for i in range(segments)]


def segments_for(radius: float) -> int:
    """Enough segments for a smooth outline at ``radius`` pixels, without wasting triangles."""
    return max(12, min(96, round(radius * 1.5)))


def fan(points: Sequence[Point]) -> list[Point]:
    """Triangles of a convex polygon."""
    first = points[0]
    triangles: list[Point] = []
    for a, b in itertools.pairwise(points[1:]):
        triangles += (first, a, b)
    return triangles


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex(points: Sequence[Point]) -> bool:
    sign = 0.0
    n = len(points)
    for i in range(n):
        turn = _cross(points[i], points[(i + 1) % n], points[(i + 2) % n])
        if turn and sign * turn < 0:
            return False
        sign = turn or sign
    return True


def _inside(p: Point, a: Point, b: Point, c: Point) -> bool:
    return _cross(a, b, p) >= 0 and _cross(b, c, p) >= 0 and _cross(c, a, p) >= 0


def earclip(points: Sequence[Point]) -> list[Point]:
    """Triangles of any simple polygon (convex or not), by clipping one "ear" at a time."""
    remaining = list(points)
    area2 = sum(_cross((0, 0), a, b) for a, b in itertools.pairwise([*remaining, remaining[0]]))
    if area2 < 0:
        remaining.reverse()  # work counterclockwise
    triangles: list[Point] = []
    while len(remaining) > 3:
        n = len(remaining)
        for i in range(n):
            a, b, c = remaining[i - 1], remaining[i], remaining[(i + 1) % n]
            if _cross(a, b, c) <= 0:
                continue  # reflex (or flat) corner: not an ear
            others = (p for p in remaining if p not in (a, b, c))
            if not any(_inside(p, a, b, c) for p in others):
                triangles += (a, b, c)
                del remaining[i]
                break
        else:
            break  # self-intersecting input: give up on the rest
    if len(remaining) == 3:
        triangles += remaining
    return triangles


def fill(points: Sequence[Point]) -> list[Point]:
    return fan(points) if is_convex(points) else earclip(points)


def _dedupe(points: Sequence[Point], closed: bool) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if not out or p != out[-1]:
            out.append(p)
    if closed and len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _normal(a: Point, b: Point) -> Point:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    return (-dy / length, dx / length)


def stroke(points: Sequence[Point], weight: float, *, closed: bool = False) -> list[Point]:
    """Triangles of a path drawn with a pen ``weight`` wide, with mitered corners."""
    pts = _dedupe(points, closed)
    if len(pts) < 2:
        return []
    half = weight / 2
    n = len(pts)
    edges = n if closed else n - 1
    normals = [_normal(pts[i], pts[(i + 1) % n]) for i in range(edges)]
    left: list[Point] = []
    right: list[Point] = []
    for i, (x, y) in enumerate(pts):
        if closed or 0 < i < n - 1:
            n0, n1 = normals[i - 1], normals[i % edges]
            mx, my = n0[0] + n1[0], n0[1] + n1[1]
            length = math.hypot(mx, my)
            if length < 1e-9:  # the path turns back on itself
                mx, my, scale = n1[0], n1[1], half
            else:
                mx, my = mx / length, my / length
                scale = min(half / max(mx * n1[0] + my * n1[1], 1e-9), MITER_LIMIT * half)
        else:
            (mx, my), scale = normals[0 if i == 0 else -1], half
        left.append((x + mx * scale, y + my * scale))
        right.append((x - mx * scale, y - my * scale))
    if closed:
        left.append(left[0])
        right.append(right[0])
    triangles: list[Point] = []
    for i in range(len(left) - 1):
        triangles += (left[i], right[i], left[i + 1], left[i + 1], right[i], right[i + 1])
    return triangles
