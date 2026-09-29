import random

import numpy as np
import pytest

from noc.ch04_particles import textures
from noc.ch04_particles.particles import (
    Confetti,
    Emitter,
    Particle,
    PointForce,
    block,
    falling,
    shatter,
    smoke,
)
from noc.common.vector import Vector


def test_particle_fades_and_dies() -> None:
    particle = Particle(Vector(), decay=2)
    for _ in range(127):
        particle.update()
    assert particle.alpha == 1
    assert not particle.is_dead
    particle.update()
    particle.update()
    assert particle.is_dead
    assert particle.alpha == 0


def test_particles_are_movers() -> None:
    particle = Particle(Vector(), mass=2)
    particle.apply_force(Vector(0, 1))
    particle.update()
    assert particle.position == Vector(0, 0.5)


def test_default_particles_start_moving_up_and_sideways() -> None:
    rng = random.Random(1)
    for _ in range(100):
        p = falling(Vector(), rng)
        assert -1 <= p.velocity.x <= 1
        assert -1 <= p.velocity.y <= 0


def test_smoke_rises() -> None:
    rng = random.Random(2)
    velocities = [smoke(Vector(), rng).velocity.y for _ in range(1000)]
    assert sum(velocities) / len(velocities) == pytest.approx(-1, abs=0.05)


def test_emitter_reaches_a_steady_population() -> None:
    emitter = Emitter(Vector(320, 50), rng=random.Random(3))
    for _ in range(400):
        emitter.add_particle()
        emitter.run()
    # A lifespan of 255 at 2 per frame lasts 128 updates; the newest has had one.
    assert len(emitter.particles) == 127


def test_add_particle_takes_a_count() -> None:
    emitter = Emitter(Vector())
    emitter.add_particle(3)
    assert len(emitter.particles) == 3


def test_budgeted_emitter_finishes() -> None:
    emitter = Emitter(Vector(), budget=10)
    for _ in range(20):
        emitter.add_particle()
    assert emitter.emitted == 10
    assert not emitter.finished
    for _ in range(200):
        emitter.run()
    assert emitter.finished


def test_forces_apply_to_every_particle() -> None:
    emitter = Emitter(Vector(), factory=lambda pos, rng: Particle(pos))
    emitter.add_particle(5)
    emitter.apply_force(Vector(1, 0))
    assert all(p.acceleration == Vector(1, 0) for p in emitter.particles)


def test_repeller_pushes_and_attractor_pulls() -> None:
    particle = Particle(Vector(0, 0))
    repel = PointForce(Vector(10, 0), 150)(particle)
    attract = PointForce(Vector(10, 0), -150)(particle)
    assert repel.xy == pytest.approx((-1.5, 0))
    assert attract.xy == pytest.approx((1.5, 0))
    far = PointForce(Vector(500, 0), 150)(particle)
    assert far.x == pytest.approx(-150 / 2500)  # distance clamped to 50


def test_confetti_is_a_particle() -> None:
    confetti = Confetti(Vector(), Vector(1, 0))
    confetti.update()
    assert isinstance(confetti, Particle)
    assert confetti.lifespan == 253


def test_block_shatters_into_fading_shards() -> None:
    shards = block(Vector(270, 70), 10, 10, 10)
    assert len(shards) == 100
    assert shards[11].position == Vector(280, 80)
    for shard in shards:
        shard.update()
    assert all(not s.is_dead and s.velocity == Vector() for s in shards)
    shatter(shards, random.Random(4))
    for shard in shards:
        shard.update()
    assert all(s.velocity.mag() == pytest.approx(10 * 0.95) for s in shards)
    assert all(s.lifespan == 253 for s in shards)


@pytest.mark.parametrize("make", [textures.blob, textures.disk, textures.ring, textures.star])
def test_textures_are_white_with_alpha(make: object) -> None:
    image = make()  # type: ignore[operator]
    pixels = np.asarray(image)
    assert pixels.shape == (32, 32, 4)
    assert (pixels[..., :3] == 255).all()
    assert pixels[0, 0, 3] == 0  # transparent corner


def test_blob_fades_from_the_center() -> None:
    alpha = np.asarray(textures.blob())[..., 3]
    assert alpha[16, 16] > alpha[16, 24] > alpha[16, 30]
