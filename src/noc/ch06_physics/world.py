"""Helpers for building pymunk worlds that behave like the book's Matter.js ones.

Matter.js works in pixels and milliseconds and integrates 16.67 ms per update; the sketches
here step pymunk one *frame* at a time, so the constants are converted to per-frame units:

- Matter's default gravity (``y = 1``, scaled by 0.001 and by the squared step) is about
  0.28 pixels per frame², and its default ``frictionAir`` of 0.01 removes 1% of the velocity
  each step, which is pymunk's ``space.damping = 0.99`` with a step of one frame.
- Matter gives bodies a density of 0.001 per square pixel; pymunk shapes take a ``density``
  too, so masses match.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pymunk

MATTER_GRAVITY = 0.28
DENSITY = 0.001
FRICTION = 0.1


def new_space(gravity: float = MATTER_GRAVITY, damping: float = 0.99) -> pymunk.Space:
    space = pymunk.Space()
    space.gravity = (0, gravity)
    space.damping = damping
    return space


def step(space: pymunk.Space, substeps: int = 4) -> None:
    """One frame, split into ``substeps`` for steadier contacts and joints."""
    for _ in range(substeps):
        space.step(1 / substeps)


def _finish(
    space: pymunk.Space, body: pymunk.Body, shapes: Sequence[pymunk.Shape], restitution: float
) -> pymunk.Body:
    for shape in shapes:
        shape.elasticity = restitution
        shape.friction = FRICTION
        if body.body_type == pymunk.Body.DYNAMIC:
            shape.density = DENSITY
    space.add(body, *shapes)
    return body


def add_box(
    space: pymunk.Space,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    restitution: float = 0.0,
    static: bool = False,
) -> pymunk.Body:
    """``Bodies.rectangle``: centered at ``(x, y)``."""
    body = pymunk.Body(body_type=pymunk.Body.STATIC if static else pymunk.Body.DYNAMIC)
    body.position = (x, y)
    return _finish(space, body, [pymunk.Poly.create_box(body, (w, h))], restitution)


def add_circle(
    space: pymunk.Space,
    x: float,
    y: float,
    r: float,
    *,
    restitution: float = 0.0,
    static: bool = False,
) -> pymunk.Body:
    """``Bodies.circle``."""
    body = pymunk.Body(body_type=pymunk.Body.STATIC if static else pymunk.Body.DYNAMIC)
    body.position = (x, y)
    return _finish(space, body, [pymunk.Circle(body, r)], restitution)


def add_polygon(
    space: pymunk.Space,
    x: float,
    y: float,
    vertices: Sequence[tuple[float, float]],
    *,
    restitution: float = 0.0,
) -> pymunk.Body:
    """``Bodies.fromVertices``: pymunk takes the convex hull of the vertices, like Matter."""
    body = pymunk.Body()
    body.position = (x, y)
    return _finish(space, body, [pymunk.Poly(body, list(vertices))], restitution)


def add_lollipop(
    space: pymunk.Space, x: float, y: float, *, restitution: float = 1.0
) -> pymunk.Body:
    """Example 6.5: a stick and a candy, two shapes on one body."""
    body = pymunk.Body()
    body.position = (x, y)
    stick = pymunk.Poly.create_box(body, (24, 4))
    candy = pymunk.Circle(body, 8, offset=(12, 0))
    return _finish(space, body, [stick, candy], restitution)


def remove(space: pymunk.Space, body: pymunk.Body) -> None:
    space.remove(body, *body.shapes, *body.constraints)


@dataclass
class MouseGrab:
    """``MouseConstraint``: a kinematic body at the mouse, pinned to whatever it grabs."""

    space: pymunk.Space
    max_force: float = 50.0
    joint: pymunk.PivotJoint | None = None

    def __post_init__(self) -> None:
        self.hand = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    def press(self, x: float, y: float) -> bool:
        hit = self.space.point_query_nearest((x, y), 0, pymunk.ShapeFilter())
        if hit is None or hit.shape is None or hit.shape.body.body_type != pymunk.Body.DYNAMIC:
            return False
        self.hand.position = (x, y)
        body = hit.shape.body
        self.joint = pymunk.PivotJoint(self.hand, body, (0, 0), body.world_to_local((x, y)))
        self.joint.max_force = self.max_force
        self.joint.error_bias = (1 - 0.3) ** 60  # stiffness, like Matter's 0.7
        self.space.add(self.joint)
        return True

    def move(self, x: float, y: float) -> None:
        self.hand.position = (x, y)

    def release(self) -> None:
        if self.joint is not None:
            self.space.remove(self.joint)
            self.joint = None
