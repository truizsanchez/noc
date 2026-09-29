"""Chapter 4's examples and exercises, in book order."""

from __future__ import annotations

import math
import random

import arcade

from noc.ch03_oscillation.oscillation import Spaceship
from noc.ch04_particles import textures
from noc.ch04_particles.particles import (
    Confetti,
    Emitter,
    Particle,
    PointForce,
    TexturedParticle,
    block,
    falling,
    shatter,
    smoke,
)
from noc.common.mathutils import remap
from noc.common.physics import Mover
from noc.common.vector import Vector
from noc.common.view import Canvas, Color, Sketch, gray, make_texture

GRAVITY = Vector(0, 0.05)


def draw_particle(canvas: Canvas, particle: Particle) -> None:
    """Circles for plain particles, spinning squares for confetti (Example 4.5)."""
    a = particle.alpha
    match particle:
        case Confetti(position=position):
            angle = remap(position.x, 0, canvas.width, 0, 4 * math.pi)
            with canvas.pushed():
                canvas.translate(*position.xy)
                canvas.rotate(angle)
                canvas.rect(
                    0, 0, 12, 12, fill=gray(127, a), stroke=gray(0, a), weight=2, center=True
                )
        case Particle(position=position):
            canvas.circle(*position.xy, 8, fill=gray(127, a), stroke=gray(0, a), weight=2)


def draw_emitter(canvas: Canvas, emitter: Emitter) -> None:
    for particle in emitter.particles:
        draw_particle(canvas, particle)


class SingleParticle(Sketch):
    title = "Example 4.1: A Single Particle"

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random()
        self.particle = falling(Vector(self.canvas.width / 2, 10), self.rng)

    def step(self) -> None:
        self.particle.apply_force(Vector(0, 0.1))
        self.particle.update()
        if self.particle.is_dead:  # start over
            self.particle = falling(Vector(self.canvas.width / 2, 10), self.rng)

    def draw(self, canvas: Canvas) -> None:
        draw_particle(canvas, self.particle)


class ArrayOfParticles(Sketch):
    title = "Example 4.2: An Array of Particles"

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random()
        self.particles: list[Particle] = []

    def step(self) -> None:
        self.particles.append(falling(Vector(self.canvas.width / 2, 20), self.rng))
        for particle in self.particles:
            particle.apply_force(GRAVITY)
            particle.update()
        self.particles = [p for p in self.particles if not p.is_dead]

    def draw(self, canvas: Canvas) -> None:
        for particle in self.particles:
            draw_particle(canvas, particle)


class EmitterSketch(Sketch):
    """One emitter adding a particle per frame, with gravity on every particle."""

    gravity = GRAVITY

    def __init__(self) -> None:
        super().__init__()
        self.emitter = Emitter(Vector(self.canvas.width / 2, 50))

    def step(self) -> None:
        self.emitter.add_particle()
        self.emitter.apply_force(self.gravity)
        self.emitter.run()

    def draw(self, canvas: Canvas) -> None:
        draw_emitter(canvas, self.emitter)


class SingleEmitter(EmitterSketch):
    title = "Example 4.3: A Single Particle Emitter"


class MovingEmitter(EmitterSketch):
    title = "Exercise 4.3: An Emitter That Follows the Mouse"
    help = ("Move the mouse",)

    def step(self) -> None:
        self.emitter.origin = Vector(*self.mouse)
        super().step()


class AsteroidsThrusters(Sketch):
    title = "Exercise 4.4: Asteroids with Thruster Particles"
    help = ("Left/Right: turn   Up: thrust",)
    ship_size = 16

    def __init__(self) -> None:
        super().__init__()
        center = Vector(self.canvas.width / 2, self.canvas.height / 2)
        self.ship = Spaceship(Mover(center, top_speed=6))
        self.exhaust = Emitter(center, self.exhaust_particle)

    def exhaust_particle(self, position: Vector, rng: random.Random) -> Particle:
        # Pushed out the back, opposite the thrust, with a little random spread.
        backward = Vector.from_angle(self.ship.heading + math.pi / 2, 5)
        return Particle(position, Vector.random2d(rng), acceleration=backward)

    def step(self) -> None:
        ship = self.ship
        if arcade.key.LEFT in self.held:
            ship.turn(-0.03)
        elif arcade.key.RIGHT in self.held:
            ship.turn(0.03)
        ship.thrusting = arcade.key.UP in self.held
        if ship.thrusting:
            ship.thrust()
            self.exhaust.origin = ship.body.position
            self.exhaust.add_particle()
        ship.update(self.canvas.width, self.canvas.height, self.ship_size * 2)
        self.exhaust.run()

    def draw(self, canvas: Canvas) -> None:
        for particle in self.exhaust.particles:
            canvas.circle(*particle.position.xy, 12, fill=(127, 0, 0, particle.alpha), stroke=None)
        r = self.ship_size
        canvas.translate(*self.ship.body.position.xy)
        canvas.rotate(self.ship.heading)
        flame = (255, 0, 0) if self.ship.thrusting else gray(175)
        canvas.rect(-r / 2, r, r / 3, r / 2, fill=flame, weight=2, center=True)
        canvas.rect(r / 2, r, r / 3, r / 2, fill=flame, weight=2, center=True)
        canvas.polygon([(-r, r), (0, -r), (r, r)], fill=gray(175), weight=2)


class SystemOfSystems(Sketch):
    title = "Example 4.4: A System of Systems"
    help = ("Click to add an emitter",)
    budget: int | None = None

    def __init__(self) -> None:
        super().__init__()
        self.emitters: list[Emitter] = []

    def mouse_down(self) -> None:
        self.emitters.append(Emitter(Vector(*self.mouse), budget=self.budget))

    def step(self) -> None:
        for emitter in self.emitters:
            emitter.add_particle()
            emitter.apply_force(GRAVITY)
            emitter.run()
        self.emitters = [e for e in self.emitters if not e.finished]
        self.status = f"{len(self.emitters)} emitters"

    def draw(self, canvas: Canvas) -> None:
        for emitter in self.emitters:
            draw_emitter(canvas, emitter)


class LimitedEmitters(SystemOfSystems):
    title = "Exercise 4.5: Emitters That Run Out"
    help = ("Click to add an emitter of 100 particles",)
    budget = 100


class Shatter(Sketch):
    title = "Exercise 4.6: Shattering Blocks"
    help = ("Click a block to shatter it",)

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random()
        self.blocks = [block(Vector(x, 70), 10, 10, 10) for x in (60, 270, 480)]
        self.broken: set[int] = set()

    def mouse_down(self) -> None:
        x, y = self.mouse
        for i, shards in enumerate(self.blocks):
            corner = shards[0].position
            if i not in self.broken and 0 <= x - corner.x <= 100 and 0 <= y - corner.y <= 100:
                shatter(shards, self.rng)
                self.broken.add(i)

    def step(self) -> None:
        for shards in self.blocks:
            for shard in shards:
                shard.update()

    def draw(self, canvas: Canvas) -> None:
        for shards in self.blocks:
            for shard in shards:
                if not shard.is_dead:
                    fill = gray(0, min(shard.alpha, 255))
                    canvas.rect(*shard.position.xy, shard.size, shard.size, fill=fill, stroke=None)


class InheritanceAndPolymorphism(EmitterSketch):
    title = "Example 4.5: A Particle System with Inheritance and Polymorphism"

    def __init__(self) -> None:
        super().__init__()
        self.emitter = Emitter(Vector(self.canvas.width / 2, 20), self.particle_or_confetti)

    @staticmethod
    def particle_or_confetti(position: Vector, rng: random.Random) -> Particle:
        velocity = Vector(rng.uniform(-1, 1), rng.uniform(-1, 0))
        kind = Particle if rng.random() < 0.5 else Confetti
        return kind(position, velocity)


class SystemWithForces(EmitterSketch):
    title = "Example 4.6: A Particle System with Forces"
    background = None
    gravity = Vector(0, 0.1)

    def draw(self, canvas: Canvas) -> None:
        canvas.background(gray(255, 30))
        super().draw(canvas)


class SystemWithRepeller(EmitterSketch):
    title = "Example 4.7: A Particle System with a Repeller"
    gravity = Vector(0, 0.1)

    def __init__(self) -> None:
        super().__init__()
        self.emitter.origin = Vector(self.canvas.width / 2, 60)
        self.repeller = PointForce(Vector(self.canvas.width / 2, 240))

    def step(self) -> None:
        self.emitter.add_particle()
        self.emitter.apply_force(self.gravity)
        self.emitter.apply(self.repeller)
        self.emitter.run()

    def draw(self, canvas: Canvas) -> None:
        super().draw(canvas)
        canvas.circle(*self.repeller.position.xy, 32, fill=gray(127), weight=2)


class AttractorsAndRepellers(SystemWithRepeller):
    title = "Exercise 4.9: Several Attractors and Repellers"
    help = ("Dark: repellers, light: attractors",)

    def __init__(self) -> None:
        super().__init__()
        self.forces = [
            PointForce(Vector(200, 180), 150),
            PointForce(Vector(440, 180), 150),
            PointForce(Vector(160, 110), -60),
            PointForce(Vector(480, 110), -60),
        ]

    def step(self) -> None:
        self.emitter.add_particle()
        self.emitter.apply_force(self.gravity)
        for force in self.forces:
            self.emitter.apply(force)
        self.emitter.run()

    def draw(self, canvas: Canvas) -> None:
        draw_emitter(canvas, self.emitter)
        for force in self.forces:
            fill = gray(60) if force.power > 0 else gray(220)
            canvas.circle(*force.position.xy, 32, fill=fill, weight=2)


def draw_arrow(canvas: Canvas, origin: Vector, v: Vector, scale: float, color: Color) -> None:
    """The book's ``drawVector()``: an arrow for ``v`` scaled by ``scale``."""
    with canvas.pushed():
        canvas.translate(*origin.xy)
        canvas.rotate(v.heading())
        length = v.mag() * scale
        canvas.line(0, 0, length, 0, color)
        canvas.line(length, 0, length - 4, 2, color)
        canvas.line(length, 0, length - 4, -2, color)


class TextureSketch(Sketch):
    """Smoke-like systems: textured particles on black, blown by wind from the mouse."""

    help = ("Move the mouse left or right for wind",)
    per_frame = 1
    additive = False

    def __init__(self) -> None:
        super().__init__()
        self.emitter = Emitter(Vector(self.canvas.width / 2, self.canvas.height - 75), smoke)
        self.texture = make_texture(textures.blob(), "blob")

    def wind(self) -> Vector:
        return Vector(remap(self.mouse[0], 0, self.canvas.width, -0.2, 0.2), 0)

    def step(self) -> None:
        self.emitter.apply_force(self.wind())
        self.emitter.run()
        self.emitter.add_particle(self.per_frame)

    def tint(self, particle: Particle) -> Color:
        return (255, 255, 255, particle.alpha)

    def draw(self, canvas: Canvas) -> None:
        canvas.background(gray(0))
        canvas.set_additive(self.additive)
        for particle in self.emitter.particles:
            canvas.image(self.texture, *particle.position.xy, 32, tint=self.tint(particle))
        canvas.set_additive(False)
        draw_arrow(canvas, Vector(canvas.width / 2, 50), self.wind(), 500, gray(255))


class ImageTextureSystem(TextureSketch):
    title = "Example 4.8: An Image-Texture Particle System"


class Fire(TextureSketch):
    title = "Exercise 4.11: Fire"
    per_frame = 4
    additive = True

    def __init__(self) -> None:
        super().__init__()
        self.emitter.origin = Vector(self.canvas.width / 2, self.canvas.height - 30)

    def tint(self, particle: Particle) -> Color:
        # Young particles are yellow; they turn red and fade as they burn out.
        life = particle.alpha / 100
        return (255, int(60 + 180 * life**2), int(40 * life), int(120 * life))


class TextureArray(Sketch):
    title = "Exercise 4.12: An Array of Particle Textures"
    help = ("Move the mouse",)

    def __init__(self) -> None:
        super().__init__()
        images = {"blob": textures.blob(), "ring": textures.ring(), "star": textures.star()}
        self.textures = [make_texture(image, name) for name, image in images.items()]
        self.emitter = Emitter(Vector(), self.spark)

    def spark(self, position: Vector, rng: random.Random) -> Particle:
        texture = rng.randrange(len(self.textures))
        return TexturedParticle(position, Vector.random2d(rng), decay=5.0, texture=texture)

    def step(self) -> None:
        self.emitter.origin = Vector(*self.mouse)
        self.emitter.add_particle()
        self.emitter.apply_force(Vector(0, -0.2))
        self.emitter.run()

    def draw(self, canvas: Canvas) -> None:
        canvas.background(gray(0))
        canvas.set_additive(True)
        for particle in self.emitter.particles:
            if isinstance(particle, TexturedParticle):
                a = particle.alpha
                texture = self.textures[particle.texture]
                canvas.image(texture, *particle.position.xy, 32, tint=(a, a, a, 255))


class AdditiveBlending(TextureSketch):
    title = "Example 4.9: Additive Blending"
    per_frame = 3
    additive = True

    def __init__(self) -> None:
        super().__init__()
        self.emitter.origin = Vector(self.canvas.width / 2, self.canvas.height - 45)
        self.emitter.factory = self.purple_smoke

    @staticmethod
    def purple_smoke(position: Vector, rng: random.Random) -> Particle:
        velocity = Vector(rng.gauss(0, 0.3), rng.gauss(-1, 0.3))
        return Particle(position, velocity, lifespan=100.0, decay=2.5)

    def tint(self, particle: Particle) -> Color:
        return (255, 100, 255, particle.alpha)


SKETCHES: tuple[type[Sketch], ...] = (
    SingleParticle,
    ArrayOfParticles,
    SingleEmitter,
    MovingEmitter,
    AsteroidsThrusters,
    SystemOfSystems,
    LimitedEmitters,
    Shatter,
    InheritanceAndPolymorphism,
    SystemWithForces,
    SystemWithRepeller,
    AttractorsAndRepellers,
    ImageTextureSystem,
    Fire,
    TextureArray,
    AdditiveBlending,
)
