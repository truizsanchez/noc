"""Chapter 5's examples and exercises, in book order."""

from __future__ import annotations

import math
import random
from typing import ClassVar

import arcade

from noc.ch05_steering.flocking import BinLattice, Flock, QuadTree, Rect, SinCosTable
from noc.ch05_steering.vehicle import (
    FlockWeights,
    FlowField,
    Path,
    PathFollowing,
    Vehicle,
    Wander,
    arrive,
    evade,
    flee,
    flock,
    follow_field,
    follow_path,
    predicted,
    pursue,
    seek,
    separate,
    stay_within,
)
from noc.common.mathutils import clamp
from noc.common.noise import Noise
from noc.common.vector import Vector
from noc.common.view import Canvas, Color, Sketch, gray

GRAY = gray(127)


def draw_vehicle(canvas: Canvas, vehicle: Vehicle, fill: Color = GRAY) -> None:
    """A triangle pointing along the velocity."""
    r = vehicle.r
    with canvas.pushed():
        canvas.translate(*vehicle.position.xy)
        canvas.rotate(vehicle.velocity.heading())
        canvas.polygon([(r * 2, 0), (-r * 2, -r), (-r * 2, r)], fill=fill, weight=2)


def draw_target(canvas: Canvas, target: Vector) -> None:
    canvas.circle(*target.xy, 48, fill=gray(127, 100), weight=2)


class MouseVehicleSketch(Sketch):
    help = ("Move the mouse",)

    def __init__(self) -> None:
        super().__init__()
        self.vehicle = Vehicle(Vector(self.canvas.width / 2, self.canvas.height / 2), max_speed=4)

    def steering(self, target: Vector) -> Vector:
        return seek(self.vehicle, target)

    def step(self) -> None:
        self.vehicle.apply_force(self.steering(Vector(*self.mouse)))
        self.vehicle.update()

    def draw(self, canvas: Canvas) -> None:
        draw_target(canvas, Vector(*self.mouse))
        draw_vehicle(canvas, self.vehicle)


class Seeking(MouseVehicleSketch):
    title = "Example 5.1: Seeking a Target"


class Fleeing(MouseVehicleSketch):
    title = "Exercise 5.1: Fleeing from a Target"

    def step(self) -> None:
        super().step()
        self.vehicle.wrap(self.canvas.width, self.canvas.height)

    def steering(self, target: Vector) -> Vector:
        return flee(self.vehicle, target)


class PursueAndEvade(Sketch):
    title = "Exercise 5.3: Pursue and Evade"
    help = ("The triangle pursues the circle, which evades it; a catch starts over",)

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        width, height = self.canvas_size
        self.pursuer = Vehicle(Vector(width / 2, height / 2), max_speed=5, max_force=0.25, r=8)
        position = Vector(random.uniform(0, width), random.uniform(0, height))
        self.quarry = Vehicle(position, Vector.random2d() * 3, max_speed=3, max_force=0.1, r=16)

    def step(self) -> None:
        width, height = self.canvas_size
        self.pursuer.apply_force(pursue(self.pursuer, self.quarry))
        if self.quarry.position.dist(self.pursuer.position) < 120:
            self.quarry.apply_force(evade(self.quarry, self.pursuer))
        for vehicle in (self.pursuer, self.quarry):
            vehicle.update()
            vehicle.wrap(width, height)
        if self.pursuer.position.dist(self.quarry.position) < self.pursuer.r + self.quarry.r:
            self.reset()

    def draw(self, canvas: Canvas) -> None:
        canvas.circle(*predicted(self.quarry, 10).xy, 16, fill=None, stroke=gray(0, 80))
        canvas.circle(*self.quarry.position.xy, 32, fill=GRAY, weight=2)
        draw_vehicle(canvas, self.pursuer)


class Arriving(MouseVehicleSketch):
    title = "Example 5.2: Arriving at a Target"

    def steering(self, target: Vector) -> Vector:
        return arrive(self.vehicle, target)


class Wandering(Sketch):
    title = "Exercise 5.4: Wandering"
    help = ("Click to show or hide the wander circle",)

    def __init__(self) -> None:
        super().__init__()
        center = Vector(self.canvas.width / 2, self.canvas.height / 2)
        self.vehicle = Vehicle(center, Vector(1, 0), max_speed=2, max_force=0.05)
        self.wander = Wander()
        self.debug = True
        self.circle: tuple[Vector, Vector] = (center, center)

    def mouse_down(self) -> None:
        self.debug = not self.debug

    def step(self) -> None:
        self.circle = self.wander.target(self.vehicle)
        self.vehicle.apply_force(seek(self.vehicle, self.circle[1]))
        self.vehicle.update()
        self.vehicle.wrap(self.canvas.width, self.canvas.height)

    def draw(self, canvas: Canvas) -> None:
        if self.debug:
            center, target = self.circle
            canvas.circle(*center.xy, self.wander.radius * 2, fill=None)
            canvas.circle(*target.xy, 4, fill=gray(0))
            canvas.line(*self.vehicle.position.xy, *center.xy)
            canvas.line(*center.xy, *target.xy)
        draw_vehicle(canvas, self.vehicle)


class StayWithinWalls(Sketch):
    title = "Example 5.3: “Stay Within Walls” Steering Behavior"
    help = ("Click to show or hide the walls",)
    offset = 25

    def __init__(self) -> None:
        super().__init__()
        center = Vector(self.canvas.width / 2, self.canvas.height / 2)
        self.vehicle = Vehicle(center, Vector(3, 4), max_speed=3, max_force=0.15)
        self.debug = True

    def mouse_down(self) -> None:
        self.debug = not self.debug

    def step(self) -> None:
        width, height = self.canvas_size
        self.vehicle.apply_force(stay_within(self.vehicle, width, height, self.offset))
        self.vehicle.update()

    def draw(self, canvas: Canvas) -> None:
        if self.debug:
            o = self.offset
            canvas.rect(
                o, o, canvas.width - 2 * o, canvas.height - 2 * o, fill=None, stroke=gray(150)
            )
        draw_vehicle(canvas, self.vehicle)


class FlowFieldFollowing(Sketch):
    title = "Example 5.4: Flow-Field Following"
    help = (
        "Space: show/hide the field   Click: new field",
        "S: swirl around the center (Exercise 5.6)   A: animate with noise (Exercise 5.7)",
    )
    resolution = 20

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.noise, self.z = Noise(), 0.0
        self.swirl, self.animate, self.debug = False, False, True
        self.field = self.make_field()
        self.vehicles = [
            Vehicle(
                Vector(random.uniform(0, width), random.uniform(0, height)),
                max_speed=random.uniform(2, 5),
                max_force=random.uniform(0.1, 0.5),
                r=4,
            )
            for _ in range(120)
        ]

    def make_field(self) -> FlowField:
        width, height = self.canvas_size
        if self.swirl:
            return FlowField.swirl(width, height, self.resolution)
        return FlowField.from_noise(width, height, self.resolution, self.noise, self.z)

    def mouse_down(self) -> None:
        self.noise.seed(random.randrange(10_000))
        self.field = self.make_field()

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        match symbol:
            case arcade.key.SPACE:
                self.debug = not self.debug
            case arcade.key.S:
                self.swirl = not self.swirl
                self.field = self.make_field()
            case arcade.key.A:
                self.animate = not self.animate
            case _:
                return super().on_key_press(symbol, modifiers)
        return True

    def step(self) -> None:
        if self.animate and not self.swirl:
            self.z += 0.01
            self.field = self.make_field()
        width, height = self.canvas_size
        for vehicle in self.vehicles:
            vehicle.apply_force(follow_field(vehicle, self.field))
            vehicle.update()
            vehicle.wrap(width, height)

    def draw(self, canvas: Canvas) -> None:
        if self.debug:
            res = self.resolution
            for col in range(self.field.cols):
                for row in range(self.field.rows):
                    x, y = (col + 0.5) * res, (row + 0.5) * res
                    v = Vector.from_angle(float(self.field.angles[col, row]), res / 2)
                    canvas.line(x, y, x + v.x, y + v.y, gray(175))
        for vehicle in self.vehicles:
            draw_vehicle(canvas, vehicle)


class AngleBetween(Sketch):
    title = "Exercise 5.9: The Angle Between Two Vectors"
    help = ("Move the mouse",)

    def draw(self, canvas: Canvas) -> None:
        center = Vector(canvas.width / 2, canvas.height / 2)
        v = (Vector(*self.mouse) - center).set_mag(100)
        x_axis = Vector(100, 0)
        for vector in (v, x_axis):
            with canvas.pushed():
                canvas.translate(*center.xy)
                canvas.rotate(vector.heading())
                canvas.line(0, 0, vector.mag(), 0, weight=2)
                canvas.line(vector.mag(), 0, vector.mag() - 6, 3, weight=2)
                canvas.line(vector.mag(), 0, vector.mag() - 6, -3, weight=2)
        theta = v.angle_between(x_axis)
        canvas.text(f"{math.degrees(theta):.0f} degrees", 10, 160, size=24)
        canvas.text(f"{theta:.2f} radians", 10, 195, size=24)


# --- paths ---------------------------------------------------------------------------------
def draw_path(canvas: Canvas, path: Path) -> None:
    points = [p.xy for p in path.points]
    if path.closed:
        points.append(points[0])
    canvas.polyline(points, gray(200), path.radius * 2)
    canvas.polyline(points, gray(0))


class PathObject(Sketch):
    title = "Example 5.5: Creating a Path Object"

    def draw(self, canvas: Canvas) -> None:
        draw_path(
            canvas,
            Path([Vector(0, canvas.height / 3), Vector(canvas.width, 2 * canvas.height / 3)]),
        )


class PathSketch(Sketch):
    """Two vehicles following a path, with the book's debug drawing."""

    help = ("Space: show/hide the debug drawing",)
    lookahead: ClassVar[float] = 50
    target_ahead: ClassVar[float] = 10

    def __init__(self) -> None:
        super().__init__()
        self.path = self.make_path()
        height = self.canvas.height
        self.vehicles = [
            Vehicle(Vector(0, height / 2), Vector(2, 0), max_speed=2, max_force=0.04, r=4),
            Vehicle(Vector(0, height / 2), Vector(2, 0), max_speed=3, max_force=0.1, r=4),
        ]
        self.debug = True
        self.infos: list[PathFollowing] = []

    def make_path(self) -> Path:
        width, height = self.canvas_size
        return Path([Vector(0, height / 3), Vector(width, 2 * height / 3)])

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.SPACE:
            self.debug = not self.debug
            return True
        return super().on_key_press(symbol, modifiers)

    def step(self) -> None:
        start, end = self.path.points[0], self.path.points[-1]
        self.infos = []
        for vehicle in self.vehicles:
            info = follow_path(vehicle, self.path, self.lookahead, self.target_ahead)
            self.infos.append(info)
            vehicle.apply_force(info.force)
            vehicle.update()
            if vehicle.position.x > end.x + vehicle.r:  # back to the start, same offset
                offset = vehicle.position.y - end.y
                vehicle.position = Vector(start.x - vehicle.r, start.y + offset)

    def draw(self, canvas: Canvas) -> None:
        draw_path(canvas, self.path)
        if self.debug:
            for vehicle, info in zip(self.vehicles, self.infos, strict=False):
                canvas.line(*vehicle.position.xy, *info.future.xy)
                canvas.circle(*info.future.xy, 4, fill=None)
                canvas.line(*info.future.xy, *info.normal.xy)
                canvas.circle(*info.normal.xy, 4, fill=None)
                color = (255, 0, 0) if info.force else gray(0)
                canvas.circle(*info.target.xy, 8, fill=color, stroke=None)
        for vehicle in self.vehicles:
            draw_vehicle(canvas, vehicle)


class SimplePathFollowing(PathSketch):
    title = "Example 5.6: Simple Path Following"
    lookahead = 25
    target_ahead = 25


class PathSegments(Sketch):
    title = "Example 5.7: Path Made of Multiple Line Segments"

    def draw(self, canvas: Canvas) -> None:
        w, h = canvas.width, canvas.height
        draw_path(
            canvas,
            Path([Vector(-20, h / 2), Vector(100, 50), Vector(400, 200), Vector(w + 20, h / 2)]),
        )


class PathFollowingSketch(PathSketch):
    title = "Example 5.8: Path Following"
    help = ("Space: show/hide the debug drawing   Click: new path",)

    def make_path(self) -> Path:
        w, h = self.canvas_size
        return Path(
            [
                Vector(-20, h / 2),
                Vector(random.uniform(0, w / 2), random.uniform(0, h)),
                Vector(random.uniform(w / 2, w), random.uniform(0, h)),
                Vector(w + 20, h / 2),
            ]
        )

    def mouse_down(self) -> None:
        self.path = self.make_path()


# --- group behaviors -----------------------------------------------------------------------
class Separation(Sketch):
    title = "Example 5.9: Separation"
    help = ("Drag the mouse to add vehicles",)
    count = 25

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.vehicles = [
            self.new_vehicle(random.uniform(0, width), random.uniform(0, height))
            for _ in range(self.count)
        ]

    def new_vehicle(self, x: float, y: float) -> Vehicle:
        return Vehicle(Vector(x, y), max_speed=3, max_force=0.2, r=12)

    def step(self) -> None:
        if self.mouse_pressed:
            self.vehicles.append(self.new_vehicle(*self.mouse))
        width, height = self.canvas_size
        for vehicle in self.vehicles:
            vehicle.apply_force(self.steering(vehicle))
            vehicle.update()
            vehicle.wrap(width, height)

    def steering(self, vehicle: Vehicle) -> Vector:
        return separate(vehicle, self.vehicles, vehicle.r * 2)

    def draw(self, canvas: Canvas) -> None:
        for vehicle in self.vehicles:
            canvas.circle(*vehicle.position.xy, vehicle.r * 2, fill=GRAY, weight=2)


class SeekAndSeparate(Separation):
    title = "Example 5.10: Combining Steering Behaviors (Seek and Separate)"
    help = ("Move the mouse",)
    count = 50

    def new_vehicle(self, x: float, y: float) -> Vehicle:
        return Vehicle(Vector(x, y), max_speed=3, max_force=0.2, r=6)

    def steering(self, vehicle: Vehicle) -> Vector:
        return (
            separate(vehicle, self.vehicles, vehicle.r * 2) * 1.5
            + seek(vehicle, Vector(*self.mouse)) * 0.5
        )

    def step(self) -> None:
        width, height = self.canvas_size
        for vehicle in self.vehicles:
            vehicle.apply_force(self.steering(vehicle))
            vehicle.update()
            vehicle.wrap(width, height)


class CrowdPathFollowing(Sketch):
    title = "Exercise 5.13: Crowd Path Following (Separation + Path Following)"
    help = ("Click to add a vehicle",)

    def __init__(self) -> None:
        super().__init__()
        w, h = self.canvas_size
        o = 30
        self.path = Path(
            [
                Vector(o, o),
                Vector(w - o, o),
                Vector(w - o, h - o),
                Vector(w / 2, h - o * 3),
                Vector(o, h - o),
            ],
            closed=True,
        )
        self.vehicles = [
            self.new_vehicle(random.uniform(0, w), random.uniform(0, h)) for _ in range(120)
        ]
        self.grid = BinLattice(24)

    def new_vehicle(self, x: float, y: float) -> Vehicle:
        return Vehicle(
            Vector(x, y), Vector.random2d(), max_speed=random.uniform(2, 4), max_force=0.3, r=4
        )

    def mouse_down(self) -> None:
        self.vehicles.append(self.new_vehicle(*self.mouse))

    def step(self) -> None:
        width, height = self.canvas_size
        self.grid.rebuild(self.vehicles)
        for vehicle in self.vehicles:
            neighbors = self.grid.neighbors(vehicle)
            vehicle.apply_force(follow_path(vehicle, self.path, 25, 25).force)
            vehicle.apply_force(separate(vehicle, neighbors, vehicle.r * 4) * 2)
            vehicle.update()
            vehicle.wrap(width, height)

    def draw(self, canvas: Canvas) -> None:
        canvas.background(gray(240))
        draw_path(canvas, self.path)
        for vehicle in self.vehicles:
            draw_vehicle(canvas, vehicle)


class Flocking(Sketch):
    title = "Example 5.11: Flocking"
    help = (
        "Drag to add boids   M: also seek the mouse (Exercise 5.16)",
        "1/Q, 2/W, 3/E: raise/lower separation, alignment, cohesion (Exercise 5.18)",
    )

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random()
        self.flock = Flock.at(self.canvas.width / 2, self.canvas.height / 2, 120, self.rng)
        self.seek_mouse = False

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        w = self.flock.weights
        changes = {
            arcade.key.KEY_1: ("separation", 0.25), arcade.key.Q: ("separation", -0.25),
            arcade.key.KEY_2: ("alignment", 0.25), arcade.key.W: ("alignment", -0.25),
            arcade.key.KEY_3: ("cohesion", 0.25), arcade.key.E: ("cohesion", -0.25),
        }  # fmt: skip
        if symbol == arcade.key.M:
            self.seek_mouse = not self.seek_mouse
        elif symbol in changes:
            name, delta = changes[symbol]
            setattr(w, name, clamp(getattr(w, name) + delta, 0, 5))
        else:
            return super().on_key_press(symbol, modifiers)
        return True

    def step(self) -> None:
        if self.mouse_pressed:
            self.flock.add(*self.mouse, self.rng)
        extra = self.flock.seek(self.mouse) * 0.5 if self.seek_mouse else None
        self.flock.update(self.flock.forces(extra), *self.canvas_size)

    def draw(self, canvas: Canvas) -> None:
        r = 3.0
        for (x, y), (vx, vy) in zip(self.flock.positions, self.flock.velocities, strict=True):
            with canvas.pushed():
                canvas.translate(float(x), float(y))
                canvas.rotate(math.atan2(vy, vx))
                canvas.polygon([(r * 2, 0), (-r * 2, -r), (-r * 2, r)], fill=GRAY)
        w = self.flock.weights
        self.status = (
            f"{len(self.flock)} boids  separation {w.separation:.2f}  "
            f"alignment {w.alignment:.2f}  cohesion {w.cohesion:.2f}"
        )


class ObjectFlockSketch(Sketch):
    """Boids as objects, each looking only at the neighbors a spatial structure returns."""

    help = ("Drag to add boids   D: show the subdivision",)
    boids_count = 300

    def __init__(self) -> None:
        super().__init__()
        width, height = self.canvas_size
        self.boids = [self.new_boid(width / 2, height / 2) for _ in range(self.boids_count)]
        self.weights = FlockWeights()
        self.debug = False
        self.checks = 0

    @staticmethod
    def new_boid(x: float, y: float) -> Vehicle:
        velocity = Vector(random.uniform(-1, 1), random.uniform(-1, 1))
        return Vehicle(Vector(x, y), velocity, max_speed=3, max_force=0.05, r=3)

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        if symbol == arcade.key.D:
            self.debug = not self.debug
            return True
        return super().on_key_press(symbol, modifiers)

    def neighbors(self, boid: Vehicle) -> list[Vehicle]:
        raise NotImplementedError

    def prepare(self) -> None:
        """Rebuild the spatial structure for this frame."""

    def step(self) -> None:
        if self.mouse_pressed:
            self.boids.append(self.new_boid(*self.mouse))
        width, height = self.canvas_size
        self.prepare()
        self.checks = 0
        for boid in self.boids:
            neighbors = self.neighbors(boid)
            self.checks += len(neighbors)
            boid.apply_force(flock(boid, neighbors, self.weights))
            boid.update()
            boid.wrap(width, height)
        brute = len(self.boids) ** 2
        self.status = f"{len(self.boids)} boids: {self.checks} neighbor checks instead of {brute}"

    def draw(self, canvas: Canvas) -> None:
        for boid in self.boids:
            draw_vehicle(canvas, boid)


class BinLatticeFlocking(ObjectFlockSketch):
    title = "Example 5.12: Bin-Lattice Spatial Subdivision"

    def __init__(self) -> None:
        super().__init__()
        self.grid = BinLattice(40)

    def prepare(self) -> None:
        self.grid.rebuild(self.boids)

    def neighbors(self, boid: Vehicle) -> list[Vehicle]:
        return self.grid.neighbors(boid)

    def draw(self, canvas: Canvas) -> None:
        if self.debug:
            res = self.grid.resolution
            for x in range(0, canvas.width, int(res)):
                canvas.line(x, 0, x, canvas.height, gray(220))
            for y in range(0, canvas.height, int(res)):
                canvas.line(0, y, canvas.width, y, gray(220))
        super().draw(canvas)


class QuadTreeFlocking(ObjectFlockSketch):
    title = "Example 5.13: Quadtree"

    def prepare(self) -> None:
        w, h = self.canvas_size
        self.tree = QuadTree(Rect(w / 2, h / 2, w / 2 + 10, h / 2 + 10))
        for boid in self.boids:
            self.tree.insert(boid)

    def neighbors(self, boid: Vehicle) -> list[Vehicle]:
        reach = self.weights.neighbor_distance
        return list(self.tree.query(Rect(boid.position.x, boid.position.y, reach, reach)))

    def draw(self, canvas: Canvas) -> None:
        if self.debug and hasattr(self, "tree"):
            for box in self.tree.boxes():
                canvas.rect(
                    box.x, box.y, box.w * 2, box.h * 2, fill=None, stroke=gray(210), center=True
                )
        super().draw(canvas)


class SinCosLookup(Sketch):
    title = "Example 5.14: Sin/Cos Lookup Table"

    def __init__(self) -> None:
        super().__init__()
        self.table = SinCosTable()

    def draw(self, canvas: Canvas) -> None:
        table = self.table
        radius = 50 + 50 * table.sin[self.frame_count % table.period]
        for degrees in range(0, 360, 5):
            i = table.index(degrees)
            x = canvas.width / 2 + radius * table.cos[i]
            y = canvas.height / 2 + radius * table.sin[i]
            canvas.point(x, y, weight=4)


SKETCHES: tuple[type[Sketch], ...] = (
    Seeking,
    Fleeing,
    PursueAndEvade,
    Arriving,
    Wandering,
    StayWithinWalls,
    FlowFieldFollowing,
    AngleBetween,
    PathObject,
    SimplePathFollowing,
    PathSegments,
    PathFollowingSketch,
    Separation,
    SeekAndSeparate,
    CrowdPathFollowing,
    Flocking,
    BinLatticeFlocking,
    QuadTreeFlocking,
    SinCosLookup,
)
