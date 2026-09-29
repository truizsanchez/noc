"""Chapter 6's examples and exercises, in book order: pymunk for Matter.js, our Verlet for
Toxiclibs.js (with pymunk versions of the soft bodies to compare)."""

from __future__ import annotations

import itertools
import math
import random
from typing import ClassVar

import arcade
import numpy as np
import pymunk

from noc.ch06_physics import integration, world
from noc.ch06_physics.soft import (
    PymunkModel,
    SoftModel,
    Topology,
    VerletModel,
    cloth,
    soft_body,
    string,
)
from noc.ch06_physics.verlet import Attraction, FloatArray, VerletPhysics
from noc.common.mathutils import clamp
from noc.common.vector import Vector
from noc.common.view import Canvas, Color, Sketch, gray

GRAY = gray(127)


def row(point: FloatArray) -> tuple[float, float]:
    """One row of a positions array as a point."""
    return (float(point[0]), float(point[1]))


def draw_body(canvas: Canvas, body: pymunk.Body, fill: Color = GRAY, weight: float = 1) -> None:
    """Draw every shape of a body where pymunk says it is (Matter's ``Render``, by hand)."""
    for shape in body.shapes:
        match shape:
            case pymunk.Circle():
                center = body.local_to_world(shape.offset)
                canvas.circle(center.x, center.y, shape.radius * 2, fill=fill, weight=weight)
                edge = body.local_to_world(shape.offset + pymunk.Vec2d(shape.radius, 0))
                canvas.line(center.x, center.y, edge.x, edge.y, weight=weight)
            case pymunk.Poly():
                points = [body.local_to_world(v) for v in shape.get_vertices()]
                canvas.polygon([(p.x, p.y) for p in points], fill=fill, weight=weight)
            case pymunk.Segment():
                a, b = body.local_to_world(shape.a), body.local_to_world(shape.b)
                canvas.line(a.x, a.y, b.x, b.y, weight=shape.radius * 2)


class SpaceSketch(Sketch):
    """A pymunk space stepped once per frame, drawing every body."""

    gravity: ClassVar[float] = world.MATTER_GRAVITY

    def __init__(self) -> None:
        super().__init__()
        self.space = world.new_space(self.gravity)
        self.rng = random.Random()
        self.build()

    def build(self) -> None:
        """Add the bodies (p5's ``setup()`` part)."""

    def before_step(self) -> None:
        """Add bodies or forces for this frame."""

    def step(self) -> None:
        self.before_step()
        world.step(self.space)
        for body in list(self.space.bodies):
            if body.position.y > self.canvas.height + 100:  # fell off: remove it
                world.remove(self.space, body)

    def draw(self, canvas: Canvas) -> None:
        for body in self.space.bodies:
            fill = gray(200) if body.body_type == pymunk.Body.STATIC else GRAY
            draw_body(canvas, body, fill, 2)

    def boundaries(self) -> None:
        """Example 6.3's two static platforms."""
        w, h = self.canvas_size
        world.add_box(self.space, w / 4, h - 5, w / 2 - 50, 10, static=True)
        world.add_box(self.space, 3 * w / 4, h - 50, w / 2 - 50, 10, static=True)

    def launch(self, body: pymunk.Body) -> None:
        """A sideways push and a spin, as the book's boxes get."""
        body.velocity = (self.rng.uniform(-5, 5), 0)
        body.angular_velocity = 0.1


# --- Matter.js examples with pymunk ---------------------------------------------------------
class DefaultRender(SpaceSketch):
    title = "Example 6.1: The Library's Own Rendering (pymunk)"
    help = ("pymunk knows every shape, so one function can draw the whole space",)

    def build(self) -> None:
        w, h = self.canvas_size
        box = world.add_box(self.space, 100, 100, 50, 50, restitution=0.75)
        for shape in box.shapes:
            shape.friction = 0.01
        box.velocity, box.angular_velocity = (5, 0), 0.1
        world.add_box(self.space, w / 2, h - 5, w, 10, static=True)


class FallingBoxes(SpaceSketch):
    title = "Exercise 6.2: Boxes Added with the Mouse, Removed When Off-Canvas"
    help = ("Hold the mouse button to add boxes",)

    def before_step(self) -> None:
        if self.mouse_pressed:
            world.add_box(self.space, *self.mouse, 16, 16)
        self.status = f"{len(self.space.bodies)} bodies"


class BoxesAndBoundaries(SpaceSketch):
    title = "Example 6.3: Falling Boxes Hitting Boundaries"

    def build(self) -> None:
        self.boundaries()

    def before_step(self) -> None:
        if self.rng.random() < 0.1:
            w, h = self.rng.uniform(8, 16), self.rng.uniform(8, 16)
            self.launch(world.add_box(self.space, self.canvas.width / 2, 50, w, h, restitution=0.6))


class PolygonShapes(SpaceSketch):
    title = "Example 6.4: Polygon Shapes"
    vertices: ClassVar = [(-10, -10), (20, -15), (15, 0), (0, 10), (-20, 15)]

    def build(self) -> None:
        self.boundaries()

    def before_step(self) -> None:
        if self.rng.random() < 0.025:
            body = world.add_polygon(
                self.space, self.canvas.width / 2, 50, self.vertices, restitution=0.2
            )
            self.launch(body)


class CompoundBodies(SpaceSketch):
    title = "Example 6.5: Multiple Shapes on One Body"

    def build(self) -> None:
        self.boundaries()
        for body in self.space.bodies:
            for shape in body.shapes:
                shape.elasticity, shape.friction = 1.0, 0.5

    def before_step(self) -> None:
        if self.rng.random() < 0.025:
            self.launch(world.add_lollipop(self.space, self.canvas.width / 2, 50))


class Pendulum(SpaceSketch):
    title = "Example 6.6: Matter.js Pendulum (a pymunk PinJoint)"

    def build(self) -> None:
        x, y, length = self.canvas.width / 2, 10, 100
        self.anchor = world.add_circle(self.space, x, y, 12, static=True)
        self.bob = world.add_circle(self.space, x + length, y - length, 12, restitution=0.6)
        self.space.add(pymunk.PinJoint(self.anchor, self.bob))

    def draw(self, canvas: Canvas) -> None:
        a, b = self.anchor.position, self.bob.position
        canvas.line(a.x, a.y, b.x, b.y, weight=2)
        super().draw(canvas)


class Bridge(SpaceSketch):
    title = "Exercise 6.5: A Bridge of Circles and Constraints"

    def build(self) -> None:
        size = 16
        xs = list(range(0, self.canvas.width + size, size))
        links = [
            world.add_circle(
                self.space, x, 50, size / 2, restitution=0.6, static=i in (0, len(xs) - 1)
            )
            for i, x in enumerate(xs)
        ]
        for a, b in itertools.pairwise(links):
            self.space.add(pymunk.PinJoint(a, b))

    def before_step(self) -> None:
        if self.rng.random() < 0.025:
            w, h = self.rng.uniform(8, 16), self.rng.uniform(8, 16)
            self.launch(
                world.add_box(self.space, self.canvas.width / 2, -50, w, h, restitution=0.6)
            )


class Windmill(SpaceSketch):
    title = "Example 6.7: Spinning Windmill (a pymunk PivotJoint)"
    motor: ClassVar[float | None] = None

    def build(self) -> None:
        x, y = self.canvas.width / 2, self.canvas.height - 50
        self.blade = world.add_box(self.space, x, y, 120, 10)
        self.space.add(pymunk.PivotJoint(self.space.static_body, self.blade, (x, y)))
        if self.motor is not None:
            self.space.add(pymunk.SimpleMotor(self.space.static_body, self.blade, self.motor))

    def before_step(self) -> None:
        if self.rng.random() < 0.05:
            x = self.canvas.width / 2 + self.rng.uniform(-60, 60)
            world.add_circle(self.space, x, 0, 8, restitution=0.6)

    def draw(self, canvas: Canvas) -> None:
        p = self.blade.position
        canvas.line(p.x, p.y, p.x, canvas.height, weight=2)
        super().draw(canvas)


class WindmillMotor(Windmill):
    title = "Exercise 6.7: A Windmill Turned by a Motor"
    motor = -0.05  # radians per frame


class MouseConstraint(SpaceSketch):
    title = "Example 6.8: MouseConstraint Demonstration"
    help = ("Drag the boxes",)

    def build(self) -> None:
        w, h = self.canvas_size
        for x, y, bw, bh in (
            (w / 2, h - 5, w, 10),
            (w / 2, 5, w, 10),
            (5, h / 2, 10, h),
            (w - 5, h / 2, 10, h),
        ):
            world.add_box(self.space, x, y, bw, bh, static=True)
        for x in (300, 400):
            world.add_box(self.space, x, h / 2, 48, 48, restitution=0.6)
        self.grab = world.MouseGrab(self.space)

    def mouse_down(self) -> None:
        self.grab.press(*self.mouse)

    def mouse_up(self) -> None:
        self.grab.release()

    def before_step(self) -> None:
        self.grab.move(*self.mouse)


class MatterAttraction(SpaceSketch):
    title = "Example 6.9: Attraction with Matter.js (pymunk)"
    gravity = 0.0
    g = 5.5  # the book's G = 0.02 in Matter's units (force scaled by 16.67 ms squared)

    def build(self) -> None:
        w, h = self.canvas_size
        self.attractor = world.add_circle(self.space, w / 2, h / 2, 32, static=True)
        self.space.damping = 1.0  # frictionAir: 0
        for _ in range(100):
            body = world.add_circle(
                self.space,
                self.rng.uniform(0, w),
                self.rng.uniform(0, h),
                self.rng.uniform(4, 8),
                restitution=1,
            )
            body.velocity = Vector.from_angle(self.rng.uniform(0, 2 * math.pi), 2).xy

    def before_step(self) -> None:
        center = self.attractor.position
        for body in self.space.bodies:
            if body is not self.attractor:
                offset = center - body.position
                distance = clamp(offset.length, 5, 25)
                force = offset.normalized() * (self.g * body.mass / distance**2)
                body.apply_force_at_world_point(force, body.position)


class CollisionEvents(SpaceSketch):
    title = "Example 6.10: Collision Events"
    PARTICLE: ClassVar[int] = 1

    def build(self) -> None:
        w, h = self.canvas_size
        world.add_box(self.space, w / 2, h - 5, w, 10, static=True)
        self.colors: dict[pymunk.Body, Color] = {}
        handler = self.space.add_collision_handler(self.PARTICLE, self.PARTICLE)
        handler.begin = self.on_collision

    def on_collision(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: object) -> bool:
        for shape in arbiter.shapes:
            self.colors[shape.body] = (self.rng.randint(100, 255), 0, self.rng.randint(100, 255))
        return True

    def before_step(self) -> None:
        if self.rng.random() < 0.05:
            body = world.add_circle(
                self.space,
                self.rng.uniform(0, self.canvas.width),
                0,
                self.rng.uniform(4, 8),
                restitution=0.6,
            )
            for shape in body.shapes:
                shape.collision_type = self.PARTICLE

    def draw(self, canvas: Canvas) -> None:
        for body in self.space.bodies:
            static = body.body_type == pymunk.Body.STATIC
            draw_body(canvas, body, gray(200) if static else self.colors.get(body, GRAY), 2)


class VanishingParticles(CollisionEvents):
    title = "Exercise 6.9: Particles That Disappear When They Collide"

    def on_collision(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: object) -> bool:
        # Bodies can't be removed while the space is stepping, so schedule it for later.
        for shape in arbiter.shapes:
            body = shape.body
            space.add_post_step_callback(self.vanish, body)
        return True

    @staticmethod
    def vanish(space: pymunk.Space, body: pymunk.Body) -> None:
        if body in space.bodies:
            world.remove(space, body)


# --- integration methods -------------------------------------------------------------------
class IntegrationMethods(Sketch):
    title = "A Brief Interlude: Integration Methods (One Step per Frame)"
    help = ("Red: explicit Euler   Green: semi-implicit Euler (the book's)   Blue: Verlet",)
    background = None
    gm = 440.0

    def __init__(self) -> None:
        super().__init__()
        self.sun = Vector(self.canvas.width / 2, self.canvas.height / 2)
        start, velocity = self.sun + Vector(100, 0), Vector(0, 2.1)
        self.euler = integration.Body(start, velocity)
        self.symplectic = integration.Body(start, velocity)
        self.verlet = integration.VerletBody.launched(start, velocity)
        self.gravity = integration.gravity_toward(self.sun, self.gm)
        self.cleared = False

    def step(self) -> None:
        integration.explicit_euler(self.euler, self.gravity)
        integration.semi_implicit_euler(self.symplectic, self.gravity)
        integration.verlet(self.verlet, self.gravity)

    def draw(self, canvas: Canvas) -> None:
        if not self.cleared:
            canvas.background(gray(255))
            self.cleared = True
        canvas.circle(*self.sun.xy, 12, fill=gray(0), stroke=None)
        for body, color in (
            (self.euler, (220, 40, 40)),
            (self.symplectic, (40, 160, 40)),
            (self.verlet, (40, 80, 220)),
        ):
            canvas.point(*body.position.xy, color, 3)
        energy = [
            integration.orbital_energy(b.position, b.velocity, self.sun, self.gm)
            for b in (self.euler, self.symplectic, self.verlet)
        ]
        self.status = "energy  " + "  ".join(f"{e:+.2f}" for e in energy)


# --- Toxiclibs.js examples with our Verlet engine ------------------------------------------
class VerletSketch(Sketch):
    """A Verlet world; one particle can be dragged with the mouse."""

    drag_index: ClassVar[int | None] = None

    def __init__(self) -> None:
        super().__init__()
        self.physics = self.build()

    def build(self) -> VerletPhysics:
        raise NotImplementedError

    def step(self) -> None:
        self.physics.update()
        if self.mouse_pressed and self.drag_index is not None:
            self.physics.move_to(self.drag_index, *self.mouse)


class SimpleSpring(VerletSketch):
    title = "Example 6.11: Simple Spring with Toxiclibs.js (Our Verlet)"
    help = ("Hold the mouse button to move the free particle",)
    drag_index = 1

    def build(self) -> VerletPhysics:
        w, h = self.canvas_size
        physics = VerletPhysics(gravity=(0, 0.5), bounds=(0, 0, w, h))
        fixed = physics.add_particle(w / 2, 0)
        free = physics.add_particle(w / 2 + 120, 0)
        physics.lock(fixed)
        physics.add_spring(fixed, free, 120, 0.01)
        return physics

    def draw(self, canvas: Canvas) -> None:
        (ax, ay), (bx, by) = self.physics.positions
        canvas.line(ax, ay, bx, by, weight=2)
        canvas.circle(ax, ay, 16, fill=GRAY, weight=2)
        canvas.circle(bx, by, 16, fill=GRAY, weight=2)


class SoftSketch(Sketch):
    """A soft body simulated by our Verlet engine or by pymunk; ``V`` switches."""

    help = ("Hold the mouse button to drag   V: switch between Verlet and pymunk",)

    def __init__(self) -> None:
        super().__init__()
        self.use_pymunk = False
        self.model: SoftModel = VerletModel(self.topology())

    def topology(self) -> Topology:
        raise NotImplementedError

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.V:
            self.use_pymunk = not self.use_pymunk
            topology = self.topology()
            self.model = PymunkModel(topology) if self.use_pymunk else VerletModel(topology)
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        self.model.step()
        if self.mouse_pressed:
            self.model.drag(self.dragged(), *self.mouse)
        self.status = f"{self.model.name}: {self.model.step_ms:.2f} ms per frame"

    def dragged(self) -> int:
        """The particle the mouse moves."""
        return 0


class SoftString(SoftSketch):
    title = "Example 6.12: Soft Swinging Pendulum"

    def topology(self) -> Topology:
        return string(*self.canvas_size)

    def dragged(self) -> int:
        return len(self.model.topology.points) - 1

    def draw(self, canvas: Canvas) -> None:
        points = self.model.points()
        canvas.polyline(points, weight=2)
        canvas.circle(*points[-1], 32, fill=GRAY, weight=2)


class Cloth(SoftSketch):
    title = "Exercise 6.10: A Hanging Cloth"
    help = ("Hold the mouse button to pull the cloth   V: switch between Verlet and pymunk",)

    def topology(self) -> Topology:
        return cloth()

    def step(self) -> None:
        self.model.step()
        if self.mouse_pressed:
            points = np.array(self.model.points())
            nearest = int(np.argmin(((points - np.array(self.mouse)) ** 2).sum(axis=1)))
            if nearest not in self.model.topology.pinned:
                self.model.drag(nearest, *self.mouse)
        self.status = f"{self.model.name}: {self.model.step_ms:.2f} ms per frame"

    def draw(self, canvas: Canvas) -> None:
        points = self.model.points()
        for s in self.model.topology.springs:
            canvas.line(*points[s.a], *points[s.b])


class SoftBodyCharacter(SoftSketch):
    title = "Example 6.13: Soft-Body Character"

    def topology(self) -> Topology:
        return soft_body(*self.canvas_size)

    def draw(self, canvas: Canvas) -> None:
        points = self.model.points()
        canvas.polygon(points, fill=gray(127, 127), weight=2)
        for s in self.model.topology.springs:
            canvas.line(*points[s.a], *points[s.b], gray(0, 60))
        # A face, so it reads as a character.
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        for dx in (-20, 20):
            canvas.circle(cx + dx, cy - 15, 12, fill=gray(255))
            canvas.circle(cx + dx, cy - 15, 4, fill=gray(0), stroke=None)


class ClusterSketch(Sketch):
    """Force-directed graphs: every node of a cluster springs to every other."""

    help = ("C: connections   P: particles   N: new graph",)

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random()
        self.show_connections, self.show_particles = True, True
        self.physics = VerletPhysics()
        self.clusters: list[list[int]] = []
        self.new_graph()

    def cluster(self, n: int, length: float) -> list[int]:
        w, h = self.canvas_size
        nodes = [
            self.physics.add_particle(
                w / 2 + self.rng.uniform(-1, 1), h / 2 + self.rng.uniform(-1, 1)
            )
            for _ in range(n)
        ]
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                self.physics.add_spring(a, b, length, 0.01)
        return nodes

    def new_graph(self) -> None:
        self.physics.clear()
        self.clusters = [
            self.cluster(self.rng.randint(2, 19), self.rng.uniform(10, self.canvas.height / 2))
        ]

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        match symbol:
            case arcade.key.C:
                self.show_connections = not self.show_connections
                self.show_particles = self.show_particles or not self.show_connections
            case arcade.key.P:
                self.show_particles = not self.show_particles
                self.show_connections = self.show_connections or not self.show_particles
            case arcade.key.N:
                self.new_graph()
            case _:
                return super().on_key_press(symbol, modifiers)
        return True

    def step(self) -> None:
        self.physics.update()

    def draw(self, canvas: Canvas) -> None:
        positions = self.physics.positions
        if self.show_connections:
            for a, b in self.physics.spring_ends:
                canvas.line(*row(positions[a]), *row(positions[b]), gray(0, 100))
        if self.show_particles:
            for x, y in positions:
                canvas.circle(float(x), float(y), 8, fill=GRAY, weight=2)


class Cluster(ClusterSketch):
    title = "Example 6.14: Cluster (a Force-Directed Graph)"

    def step(self) -> None:
        super().step()
        if self.frame_count % 120 == 119:
            self.new_graph()


class ConnectedClusters(ClusterSketch):
    title = "Exercise 6.13: Clusters Connected by Minimum-Distance Springs"
    help = ("C: connections   P: particles   N: new graph   (the view zooms to fit)",)

    def new_graph(self) -> None:
        # Drag, and fewer relaxation passes: every particle has dozens of min-distance springs,
        # and without drag their pushes keep the graph expanding forever.
        self.physics = VerletPhysics(drag=0.1, iterations=10)
        # An invisible pinned particle gently pulls everything back toward the center
        # (an AttractionBehavior, as Exercise 6.14 suggests combining with springs).
        w, h = self.canvas_size
        center = self.physics.add_particle(w / 2, h / 2)
        self.physics.lock(center)
        self.physics.attractions.append(Attraction(center, 2000, 0.05))
        lengths = [self.rng.uniform(20, 100) for _ in range(8)]
        self.clusters = [self.cluster(self.rng.randint(3, 7), length) for length in lengths]
        for i, a in enumerate(self.clusters):
            for j in range(i + 1, len(self.clusters)):
                spacing = (lengths[i] + lengths[j]) / 2
                for p in a:
                    for q in self.clusters[j]:
                        self.physics.add_spring(p, q, spacing, 0.05, min_distance=True)
        self.zoom = 1.0

    def draw(self, canvas: Canvas) -> None:
        positions = self.physics.positions
        center = positions[1:].mean(axis=0)  # particle 0 is the invisible center
        extent = np.abs(positions[1:] - center).max(axis=0) + 20
        fit = min(canvas.width / 2 / extent[0], canvas.height / 2 / extent[1], 1.0)
        self.zoom += (fit - self.zoom) * 0.1
        canvas.translate(canvas.width / 2, canvas.height / 2)
        canvas.scale(self.zoom)
        canvas.translate(-float(center[0]), -float(center[1]))
        if self.show_connections:
            ends, min_only = self.physics.spring_ends, self.physics.min_distance
            for (a, b), skip in zip(ends, min_only, strict=True):
                if not skip:
                    canvas.line(*row(positions[a]), *row(positions[b]), gray(0, 100))
        if self.show_particles:
            for k, nodes in enumerate(self.clusters):
                hue = (60 + 37 * k) % 256
                for i in nodes:
                    canvas.circle(*row(positions[i]), 8, fill=(hue, 120, 255 - hue), weight=1)
        self.status = f"zoom {self.zoom:.2f}"


class AttractionBehaviors(VerletSketch):
    title = "Example 6.15: Attraction (and Repulsion) Behaviors"
    help = ("Hold the mouse button to move the attractor",)

    def build(self) -> VerletPhysics:
        w, h = self.canvas_size
        physics = VerletPhysics(drag=0.01, bounds=(0, 0, w, h))
        for _ in range(50):
            particle = physics.add_particle(random.uniform(0, w), random.uniform(0, h))
            physics.attractions.append(Attraction(particle, 8, -2))  # personal space
        self.attractor = physics.add_particle(w / 2, h / 2)
        physics.attractions.append(Attraction(self.attractor, w, 0.1))
        physics.attractions.append(Attraction(self.attractor, 20, -5))
        return physics

    def step(self) -> None:
        self.physics.lock(self.attractor, self.mouse_pressed)
        if self.mouse_pressed:
            self.physics.move_to(self.attractor, *self.mouse)
        self.physics.update()

    def draw(self, canvas: Canvas) -> None:
        positions = self.physics.positions
        canvas.circle(*row(positions[self.attractor]), 32, fill=gray(0), stroke=None)
        for i, (x, y) in enumerate(positions):
            if i != self.attractor:
                canvas.circle(float(x), float(y), 8, fill=GRAY, weight=2)


SKETCHES: tuple[type[Sketch], ...] = (
    DefaultRender,
    FallingBoxes,
    BoxesAndBoundaries,
    PolygonShapes,
    CompoundBodies,
    Pendulum,
    Bridge,
    Windmill,
    WindmillMotor,
    MouseConstraint,
    MatterAttraction,
    CollisionEvents,
    VanishingParticles,
    IntegrationMethods,
    SimpleSpring,
    SoftString,
    Cloth,
    SoftBodyCharacter,
    Cluster,
    ConnectedClusters,
    AttractionBehaviors,
)
