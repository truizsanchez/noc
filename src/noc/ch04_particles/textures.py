"""Particle textures drawn with numpy instead of loaded from image files.

The book's smoke texture is a PNG of a white blob that fades out toward its edges; the book
mentions it can also be made in code, which is what :func:`blob` does. The other shapes stand
in for Exercise 4.12's array of images.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image


def _radius(size: int) -> npt.NDArray[np.float64]:
    """Distance from the center of a ``size x size`` image, 0 at the center, 1 at the edge."""
    coords = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(coords, coords)
    return np.hypot(x, y)


def _image(alpha: npt.NDArray[np.float64]) -> Image.Image:
    rgba = np.zeros((*alpha.shape, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgba)


def blob(size: int = 32, falloff: float = 2.0) -> Image.Image:
    """A white disk whose opacity fades smoothly to zero at its edge (Figure 4.8, right)."""
    return _image(np.clip(1 - _radius(size), 0, 1) ** falloff)


def disk(size: int = 32) -> Image.Image:
    """A plain white circle with a hard edge (Figure 4.8, left)."""
    return _image((_radius(size) <= 1).astype(np.float64))


def ring(size: int = 32, width: float = 0.25) -> Image.Image:
    """A soft ring."""
    return _image(1 - np.abs(_radius(size) - (1 - width)) / width)


def star(size: int = 32, points: int = 5) -> Image.Image:
    """A soft star with ``points`` rays."""
    coords = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(coords, coords)
    angle = np.arctan2(y, x)
    reach = 0.45 + 0.55 * np.abs(np.cos(angle * points / 2)) ** 3
    return _image(1 - np.hypot(x, y) / reach)
