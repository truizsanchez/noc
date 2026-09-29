"""Chapter 2's examples and exercises, in book order."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import arcade

from noc.ch02_forces.forces import (
    Balloon,
    Galaxy,
    attract_all,
    attract_and_repel,
    edge_repulsion,
    fan,
    figure_eight,
)
from noc.common.physics import Liquid, Mover, attraction, drag, friction, weight
from noc.common.vector import Vector
from noc.common.view import Canvas, Color, Sketch, gray

GRAVITY = Vector(0, 0.1)
WIND = Vector(0.1, 0)
MOVER_FILL = gray(127, 127)


def draw_mover(canvas: Canvas, mover: Mover, diameter: float, fill: Color = MOVER_FILL) -> None:
    canvas.circle(*mover.position.xy, diameter, fill=fill, weight=2)


class FloatingBalloon(Sketch):
    title = "Exercise 2.1: A Helium Balloon in a Perlin Noise Breeze"
    diameter = 48

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.balloon = Balloon(Mover(Vector(width / 2, height / 2), top_speed=5))

    def step(self) -> None:
        self.balloon.step(self.canvas.width, self.diameter)

    def draw(self, canvas: Canvas) -> None:
        x, y = self.balloon.body.position
        bottom = y + self.diameter / 2
        canvas.line(x, bottom + 5, x, bottom + 100)
        canvas.polygon(
            [(x, bottom), (x + 5, bottom + 5), (x - 5, bottom + 5)], fill=(255, 192, 203)
        )
        canvas.circle(x, y, self.diameter, fill=(255, 192, 203))


class Forces(Sketch):
    title = "Example 2.1: Forces"
    help = ("Hold the mouse button for wind",)

    def __init__(self) -> None:
        super().__init__()
        self.mover = Mover(Vector(self.canvas.width / 2, 30))

    def step(self) -> None:
        self.mover.apply_force(GRAVITY)
        if self.mouse_pressed:
            self.mover.apply_force(WIND)
        self.mover.update()
        self.mover.bounce(self.canvas.width, self.canvas.height)

    def draw(self, canvas: Canvas) -> None:
        draw_mover(canvas, self.mover, 48)


class TwoMoversSketch(Sketch):
    """A heavy and a light mover, pushed by the same forces."""

    help = ("Hold the mouse button for wind",)

    def __init__(self) -> None:
        super().__init__()
        self.movers = [Mover(Vector(200, 30), mass=10), Mover(Vector(440, 30), mass=2)]

    def forces(self, mover: Mover) -> list[Vector]:
        wind = [WIND] if self.mouse_pressed else []
        return [GRAVITY, *wind]

    def step(self) -> None:
        width, height = self.canvas_size
        for mover in self.movers:
            for force in self.forces(mover):
                mover.apply_force(force)
            mover.update()
            mover.bounce(width, height, self.radius(mover))

    def radius(self, mover: Mover) -> float:
        return 0.0

    def draw(self, canvas: Canvas) -> None:
        for mover in self.movers:
            draw_mover(canvas, mover, mover.mass * 16)


class ForcesOnTwoObjects(TwoMoversSketch):
    title = "Example 2.2: Forces Acting on Two Objects"


class InvisibleWalls(Sketch):
    title = "Exercise 2.3: An Invisible Force Keeping Objects Inside"
    help = ("Hold the mouse button for wind",)

    def __init__(self) -> None:
        super().__init__()
        self.movers = [
            Mover(
                Vector(random.uniform(100, 540), random.uniform(40, 200)), mass=random.uniform(1, 4)
            )
            for _ in range(8)
        ]

    def step(self) -> None:
        width, height = self.canvas_size
        for mover in self.movers:
            mover.apply_force(weight(mover.mass))
            if self.mouse_pressed:
                mover.apply_force(WIND * 3)
            mover.apply_force(edge_repulsion(mover.position, width, height, 60, 2))
            mover.update()

    def draw(self, canvas: Canvas) -> None:
        for mover in self.movers:
            draw_mover(canvas, mover, mover.mass * 16)


class Fan(Sketch):
    title = "Exercise 2.5: A Fan at the Mouse"
    help = ("Hold the mouse button to switch on the fan at the mouse",)

    def __init__(self) -> None:
        super().__init__()
        self.movers = [Mover(Vector(80 + i * 60, 30), mass=random.uniform(1, 3)) for i in range(9)]

    def step(self) -> None:
        width, height = self.canvas_size
        for mover in self.movers:
            mover.apply_force(weight(mover.mass))
            if self.mouse_pressed:
                mover.apply_force(fan(Vector(*self.mouse), mover.position, 1.5))
            mover.update()
            mover.bounce(width, height, mover.mass * 8, restitution=0.9, top=True)

    def draw(self, canvas: Canvas) -> None:
        for mover in self.movers:
            draw_mover(canvas, mover, mover.mass * 16)
        if self.mouse_pressed:
            canvas.circle(*self.mouse, 20, fill=(120, 170, 255, 120), stroke=None)


class GravityScaledByMass(TwoMoversSketch):
    title = "Example 2.3: Gravity Scaled by Mass"

    def forces(self, mover: Mover) -> list[Vector]:
        wind = [WIND] if self.mouse_pressed else []
        return [weight(mover.mass), *wind]

    def radius(self, mover: Mover) -> float:
        return mover.mass * 8


class IncludingFriction(Sketch):
    title = "Example 2.4: Including Friction"
    help = ("Hold the mouse button for wind",)

    def __init__(self) -> None:
        super().__init__()
        self.mover = Mover(Vector(self.canvas.width / 2, 30), mass=5)

    def step(self) -> None:
        width, height = self.canvas_size
        mover, radius = self.mover, self.mover.mass * 8
        mover.apply_force(Vector(0, 1))
        if self.mouse_pressed:
            mover.apply_force(Vector(0.5, 0))
        if mover.position.y > height - radius - 1:  # touching the floor
            mover.apply_force(friction(mover.velocity, 0.1))
        mover.bounce(width, height, radius, restitution=0.9)
        mover.update()

    def draw(self, canvas: Canvas) -> None:
        draw_mover(canvas, self.mover, self.mover.mass * 16)


class FluidSketch(Sketch):
    help = ("Click to start again",)
    limit_drag = False

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.liquid = Liquid(0, height / 2, width, height / 2, self.coefficient())
        self.reset()

    def coefficient(self) -> float:
        return 0.1

    def start_height(self, i: int) -> float:
        return 0.0

    def reset(self) -> None:
        self.movers = [
            Mover(Vector(40 + i * 70, self.start_height(i)), mass=random.uniform(0.5, 3))
            for i in range(9)
        ]

    def mouse_down(self) -> None:
        self.reset()

    def step(self) -> None:
        width, height = self.canvas_size
        for mover in self.movers:
            if self.liquid.contains(mover.position):
                mass = mover.mass if self.limit_drag else None
                mover.apply_force(drag(mover.velocity, self.liquid.coefficient, mass=mass))
            mover.apply_force(weight(mover.mass))
            mover.update()
            mover.bounce(width, height, mover.mass * 8, restitution=0.9)

    def draw(self, canvas: Canvas) -> None:
        liquid = self.liquid
        canvas.rect(liquid.x, liquid.y, liquid.width, liquid.height, fill=gray(220), stroke=None)
        for mover in self.movers:
            draw_mover(canvas, mover, mover.mass * 16)


class FluidResistance(FluidSketch):
    title = "Example 2.5: Fluid Resistance"


class LimitedDrag(FluidSketch):
    title = "Exercise 2.8: Drag That Can Stop but Never Reverse"
    help = ("Click to start again; D: toggle the limit",)
    limit_drag = True

    def coefficient(self) -> float:
        return 2.0  # strong enough that unlimited drag bounces the balls off the surface

    def start_height(self, i: int) -> float:
        return random.uniform(0, 80)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.D:
            self.limit_drag = not self.limit_drag
            return True
        return super().on_key_press(symbol, modifiers)

    def draw(self, canvas: Canvas) -> None:
        super().draw(canvas)
        self.status = "drag limited" if self.limit_drag else "drag not limited"


@dataclass
class Draggable:
    """A circle the mouse can grab and move (the book's ``Attractor`` interaction)."""

    body: Mover
    radius: float
    grabbed_at: Vector | None = None

    def hovered(self, mouse: Vector) -> bool:
        return self.body.position.dist(mouse) < self.radius

    def press(self, mouse: Vector) -> None:
        if self.hovered(mouse):
            self.grabbed_at = self.body.position - mouse

    def drag_to(self, mouse: Vector) -> None:
        if self.grabbed_at is not None:
            self.body.position = mouse + self.grabbed_at

    def draw(self, canvas: Canvas, mouse: Vector) -> None:
        if self.grabbed_at is not None:
            fill = gray(50)
        elif self.hovered(mouse):
            fill = gray(100)
        else:
            fill = gray(175, 200)
        canvas.circle(*self.body.position.xy, self.radius * 2, fill=fill, weight=4)


class AttractorSketch(Sketch):
    help = ("Drag the attractor",)

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.attractor = Draggable(Mover(Vector(width / 2, height / 2), mass=20), 20)
        self.movers = self.make_movers()

    def make_movers(self) -> list[Mover]:
        return [Mover(Vector(300, 50), Vector(1, 0), mass=2)]

    def force(self, mover: Mover) -> Vector:
        a = self.attractor.body
        return attraction(a.position, a.mass, mover.position, mover.mass)

    def mouse_down(self) -> None:
        self.attractor.press(Vector(*self.mouse))

    def mouse_up(self) -> None:
        self.attractor.grabbed_at = None

    def step(self) -> None:
        self.attractor.drag_to(Vector(*self.mouse))
        for mover in self.movers:
            mover.apply_force(self.force(mover))
            mover.update()

    def draw(self, canvas: Canvas) -> None:
        self.attractor.draw(canvas, Vector(*self.mouse))
        for mover in self.movers:
            draw_mover(canvas, mover, mover.mass * 16)


class Attraction(AttractorSketch):
    title = "Example 2.6: Attraction"


class AttractionWithManyMovers(AttractorSketch):
    title = "Example 2.7: Attraction with Many Movers"

    def make_movers(self) -> list[Mover]:
        width, height = self.canvas_size
        return [
            Mover(
                Vector(random.uniform(0, width), random.uniform(0, height)),
                Vector(1, 0),
                random.uniform(0.5, 3),
            )
            for _ in range(10)
        ]


class AttractAndRepel(AttractionWithManyMovers):
    title = "Exercise 2.13: Attracting from Afar, Repelling Up Close"

    def force(self, mover: Mover) -> Vector:
        a = self.attractor.body
        return attract_and_repel(a.position, a.mass * 40, mover.position, mover.mass, 70)

    def step(self) -> None:
        super().step()
        for mover in self.movers:
            mover.velocity *= 0.99  # a little damping, or they would orbit forever


class TwoBodyAttraction(Sketch):
    title = "Example 2.8: Two-Body Attraction"

    def __init__(self) -> None:
        super().__init__()
        self.bodies = [
            Mover(Vector(320, 40), Vector(1, 0), 8),
            Mover(Vector(320, 200), Vector(-1, 0), 8),
        ]

    def step(self) -> None:
        attract_all(self.bodies)
        for body in self.bodies:
            body.update()

    def draw(self, canvas: Canvas) -> None:
        for body in self.bodies:
            draw_mover(canvas, body, body.mass**0.5 * 8, gray(127, 100))


class FigureEight(Sketch):
    title = "Exercise 2.14: A Three-Body Choreography (the Figure Eight)"
    background = None

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.bodies = figure_eight(Vector(width / 2, height / 2), 150, 600)

    def step(self) -> None:
        attract_all(self.bodies, distance_range=(1, float("inf")))
        for body in self.bodies:
            body.update()

    def draw(self, canvas: Canvas) -> None:
        canvas.background(gray(255, 30))
        for body, color in zip(
            self.bodies, [(200, 50, 50), (50, 150, 50), (50, 50, 200)], strict=True
        ):
            canvas.circle(*body.position.xy, 16, fill=color, weight=2)


class NBodies(Sketch):
    title = "Example 2.9: n Bodies"

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.bodies = [
            Mover(
                Vector(random.uniform(0, width), random.uniform(0, height)),
                mass=random.uniform(0.1, 2),
            )
            for _ in range(10)
        ]

    def step(self) -> None:
        attract_all(self.bodies)
        for body in self.bodies:
            body.update()

    def draw(self, canvas: Canvas) -> None:
        for body in self.bodies:
            draw_mover(canvas, body, body.mass * 16)


class MouseAttractsBodiesRepel(NBodies):
    title = "Exercise 2.15: Attracted to the Mouse, Repelling One Another"
    help = ("Move the mouse",)

    def step(self) -> None:
        # A constant pull toward the mouse (like Example 1.10) and inverse-square repulsion
        # between bodies: the pull gathers them, the repulsion spaces them out, and a little
        # damping lets them settle into a cluster.
        mouse = Vector(*self.mouse)
        for body in self.bodies:
            body.apply_force((mouse - body.position).set_mag(0.2 * body.mass))
            for other in self.bodies:
                if other is not body:
                    push = attraction(
                        other.position, other.mass * 100, body.position, body.mass,
                        distance_range=(10, math.inf),
                    )  # fmt: skip
                    body.apply_force(-push)
            body.velocity *= 0.95
            body.update()


class SpiralGalaxy(Sketch):
    title = "Exercise 2.16: A Spiral Galaxy"
    background = None

    def __init__(self) -> None:
        super().__init__()
        self.galaxy = Galaxy.spiral(random.Random())

    def step(self) -> None:
        self.galaxy.step()

    def draw(self, canvas: Canvas) -> None:
        canvas.background(gray(255, 50))
        canvas.translate(canvas.width / 2, canvas.height / 2)
        galaxy = self.galaxy
        for (x, y), mass in zip(galaxy.positions, galaxy.masses, strict=True):
            canvas.circle(float(x), float(y), 2 * float(mass) ** 0.5, fill=gray(0, 100), weight=2)


SKETCHES: tuple[type[Sketch], ...] = (
    FloatingBalloon,
    Forces,
    ForcesOnTwoObjects,
    InvisibleWalls,
    Fan,
    GravityScaledByMass,
    IncludingFriction,
    FluidResistance,
    LimitedDrag,
    Attraction,
    AttractionWithManyMovers,
    AttractAndRepel,
    TwoBodyAttraction,
    FigureEight,
    NBodies,
    MouseAttractsBodiesRepel,
    SpiralGalaxy,
)
