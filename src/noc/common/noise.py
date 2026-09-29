"""p5.js's ``noise()``: smooth pseudorandom values from 0 to 1 in one, two or three dimensions.

As the book points out, p5's ``noise()`` isn't Ken Perlin's gradient noise but *value noise*:
a table of 4096 random values is sampled at the lattice points around ``(x, y, z)`` and
blended with a cosine curve. Several octaves are added up, each at twice the frequency and
``falloff`` times the amplitude of the previous one (``noiseDetail()``).

This is a line-for-line port of that algorithm, so a given seed produces the same values the
book's sketches get from ``noiseSeed()``. :meth:`Noise.grid` computes a whole image at once
with numpy, for the 2D noise examples that color every pixel.
"""

from __future__ import annotations

import math
import random

import numpy as np
import numpy.typing as npt

YWRAPB = 4
YWRAP = 1 << YWRAPB
ZWRAPB = 8
ZWRAP = 1 << ZWRAPB
SIZE = 4095  # the table has SIZE + 1 entries; indices wrap with "& SIZE"

type FloatArray = npt.NDArray[np.float64]


def scaled_cosine(t: float) -> float:
    """Eases ``t`` from 0 to 1 along half a cosine: flat ends join neighboring cells smoothly."""
    return 0.5 * (1.0 - math.cos(t * math.pi))


def lcg_values(seed: int, count: int) -> list[float]:
    """p5's ``noiseSeed()`` generator: a linear congruential generator (Numerical Recipes)."""
    m, a, c = 2**32, 1664525, 1013904223
    z = seed % m
    values = []
    for _ in range(count):
        z = (a * z + c) % m
        values.append(z / m)
    return values


class Noise:
    """One noise space. Call it like p5's ``noise(x, y, z)``."""

    def __init__(self, seed: int | None = None, octaves: int = 4, falloff: float = 0.5) -> None:
        self.octaves = octaves
        self.falloff = falloff
        self.seed(seed)

    def seed(self, seed: int | None = None) -> None:
        """p5's ``noiseSeed()``; without a seed, a fresh random table (p5's default)."""
        if seed is None:
            self._table = [random.random() for _ in range(SIZE + 1)]
        else:
            self._table = lcg_values(seed, SIZE + 1)
        self._array = np.array(self._table)

    def detail(self, octaves: int, falloff: float | None = None) -> None:
        """p5's ``noiseDetail()``: non-positive arguments leave the setting unchanged."""
        if octaves > 0:
            self.octaves = octaves
        if falloff is not None and falloff > 0:
            self.falloff = falloff

    def __call__(self, x: float, y: float = 0.0, z: float = 0.0) -> float:
        table = self._table
        x, y, z = abs(x), abs(y), abs(z)
        xi, yi, zi = math.floor(x), math.floor(y), math.floor(z)
        xf, yf, zf = x - xi, y - yi, z - zi
        result = 0.0
        amplitude = 0.5
        for _ in range(self.octaves):
            of = xi + (yi << YWRAPB) + (zi << ZWRAPB)
            rxf, ryf = scaled_cosine(xf), scaled_cosine(yf)
            # Blend the four corners of the cell at z = zi, along x and then along y...
            n1 = table[of & SIZE]
            n1 += rxf * (table[(of + 1) & SIZE] - n1)
            n2 = table[(of + YWRAP) & SIZE]
            n2 += rxf * (table[(of + YWRAP + 1) & SIZE] - n2)
            n1 += ryf * (n2 - n1)
            # ...then the four at z = zi + 1, and finally along z.
            of += ZWRAP
            n2 = table[of & SIZE]
            n2 += rxf * (table[(of + 1) & SIZE] - n2)
            n3 = table[(of + YWRAP) & SIZE]
            n3 += rxf * (table[(of + YWRAP + 1) & SIZE] - n3)
            n2 += ryf * (n3 - n2)
            n1 += scaled_cosine(zf) * (n2 - n1)

            result += n1 * amplitude
            amplitude *= self.falloff
            # Next octave: twice the frequency.
            xi, xf = (xi << 1) + (xf >= 0.5), 2 * xf - (xf >= 0.5)
            yi, yf = (yi << 1) + (yf >= 0.5), 2 * yf - (yf >= 0.5)
            zi, zf = (zi << 1) + (zf >= 0.5), 2 * zf - (zf >= 0.5)
        return result

    def grid(self, xs: FloatArray, ys: FloatArray, z: float = 0.0) -> FloatArray:
        """Noise at every ``(x, y)`` pair of the two arrays (e.g. from ``np.meshgrid``)."""
        table = self._array
        x, y = np.abs(xs), np.abs(ys)
        xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        xf, yf = x - xi, y - yi
        zi, zf = math.floor(abs(z)), abs(z) - math.floor(abs(z))
        result = np.zeros_like(x)
        amplitude = 0.5
        for _ in range(self.octaves):
            of = xi + (yi << YWRAPB) + (zi << ZWRAPB)
            rxf = 0.5 * (1.0 - np.cos(xf * np.pi))
            ryf = 0.5 * (1.0 - np.cos(yf * np.pi))
            n1 = table[of & SIZE]
            n1 = n1 + rxf * (table[(of + 1) & SIZE] - n1)
            n2 = table[(of + YWRAP) & SIZE]
            n2 = n2 + rxf * (table[(of + YWRAP + 1) & SIZE] - n2)
            n1 = n1 + ryf * (n2 - n1)
            of = of + ZWRAP
            n2 = table[of & SIZE]
            n2 = n2 + rxf * (table[(of + 1) & SIZE] - n2)
            n3 = table[(of + YWRAP) & SIZE]
            n3 = n3 + rxf * (table[(of + YWRAP + 1) & SIZE] - n3)
            n2 = n2 + ryf * (n3 - n2)
            n1 = n1 + scaled_cosine(zf) * (n2 - n1)

            result += n1 * amplitude
            amplitude *= self.falloff
            xcarry, ycarry = xf >= 0.5, yf >= 0.5
            xi, xf = (xi << 1) + xcarry, 2 * xf - xcarry
            yi, yf = (yi << 1) + ycarry, 2 * yf - ycarry
            zcarry = zf >= 0.5
            zi, zf = (zi << 1) + zcarry, 2 * zf - zcarry
        return result


noise = Noise()
"""A shared noise space, like p5's global ``noise()``."""
