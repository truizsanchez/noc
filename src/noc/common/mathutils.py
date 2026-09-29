"""Small numeric helpers the sketches use constantly (p5's ``map()`` and ``constrain()``)."""


def remap(value: float, start1: float, stop1: float, start2: float, stop2: float) -> float:
    """p5's ``map()``: re-maps ``value`` from one range to another, without clamping.

    Named ``remap`` because ``map`` is a Python builtin.
    """
    return start2 + (stop2 - start2) * (value - start1) / (stop1 - start1)


def clamp(value: float, low: float, high: float) -> float:
    """p5's ``constrain()``."""
    return max(low, min(high, value))
