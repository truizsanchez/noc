"""2D noise as an image (Figure 0.9, Exercises 0.8-0.9) and as a landscape (Exercise 0.10)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from noc.common.noise import Noise

type FloatArray = npt.NDArray[np.float64]
type ByteArray = npt.NDArray[np.uint8]


def noise_field(
    noise: Noise, cols: int, rows: int, step: tuple[float, float], z: float = 0.0
) -> FloatArray:
    """Noise values for a ``rows x cols`` grid, advancing ``xoff``/``yoff`` by ``step`` per cell."""
    xs, ys = np.meshgrid(np.arange(cols) * step[0], np.arange(rows) * step[1])
    return noise.grid(xs, ys, z)


def grayscale(values: FloatArray) -> ByteArray:
    """Noise from 0 to 1 as an RGB image, 0 black and 1 white."""
    gray = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def hue_and_brightness(values: FloatArray) -> ByteArray:
    """The same value picks both hue and brightness, at full saturation (Exercise 0.8).

    A vectorized HSB-to-RGB conversion (p5's ``colorMode(HSB)``). The sketch maps noise to
    a brightness from 0 to 255, but HSB brightness tops out at 100, so p5 clamps everything
    above 0.39 to full brightness; this does the same to match the book's colors.
    """
    h = np.clip(values, 0, 1) * 6  # hue sector, 0 to 6
    v = np.clip(values * 2.55, 0, 1)
    sector = np.floor(h).astype(np.int64) % 6
    f = h - np.floor(h)
    zero = np.zeros_like(v)
    q, t = v * (1 - f), v * f
    channels = [
        (v, t, zero),
        (q, v, zero),
        (zero, v, t),
        (zero, q, v),
        (t, zero, v),
        (v, zero, q),
    ]
    rgb = np.zeros((*values.shape, 3))
    for i, (r, g, b) in enumerate(channels):
        mask = sector == i
        rgb[mask] = np.stack([r[mask], g[mask], b[mask]], axis=1)
    return (rgb * 255).astype(np.uint8)


@dataclass
class Terrain:
    """Exercise 0.10: a grid of elevations from 2D noise, scrolling through the third dimension."""

    noise: Noise
    scale: float = 20.0  # size of a cell
    width: float = 800.0
    height: float = 400.0
    zoff: float = 0.0
    elevations: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.cols = int(self.width // self.scale)
        self.rows = int(self.height // self.scale)
        self.calculate()

    def calculate(self) -> None:
        values = noise_field(self.noise, self.cols, self.rows, (0.1, 0.1), self.zoff)
        self.elevations = values.T * 240 - 120  # [col][row], from -120 to 120
        self.zoff += 0.01

    def vertices(self) -> FloatArray:
        """Every grid point as ``(x, y, elevation)``, centered on the origin: (cols, rows, 3)."""
        i, j = np.meshgrid(np.arange(self.cols), np.arange(self.rows), indexing="ij")
        x = i * self.scale - self.width / 2
        y = j * self.scale - self.height / 2
        return np.stack([x, y, self.elevations], axis=2)


def project(points: FloatArray, theta: float, width: float, height: float) -> FloatArray:
    """Where p5's WEBGL renderer puts 3D points on a ``width x height`` canvas.

    Applies Exercise 0.10's ``translate(0, 20, -200); rotateX(PI / 3); rotateZ(theta)`` and
    p5's default camera: at distance ``(height / 2) / tan(PI / 6)`` looking at the origin,
    with a 60-degree field of view. Returns ``(x, y, depth)``; larger depth is farther away.
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    c, s = math.cos(theta), math.sin(theta)
    x, y = x * c - y * s, x * s + y * c  # rotateZ(theta)
    a = math.pi / 3
    y, z = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)  # rotateX
    y, z = y + 20, z - 200  # translate
    eye = (height / 2) / math.tan(math.pi / 6)
    factor = eye / (eye - z)
    return np.stack([width / 2 + x * factor, height / 2 + y * factor, eye - z], axis=-1)
