from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectoryPoint:
    position: FloatArray
    velocity: FloatArray


def quintic_joint_trajectory(
    start: FloatArray,
    goal: FloatArray,
    time: float,
    duration: float,
) -> TrajectoryPoint:
    """Interpolate position and velocity with zero endpoint velocity/acceleration."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    phase = float(np.clip(time / duration, 0.0, 1.0))
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    blend_rate = (30.0 * phase**2 - 60.0 * phase**3 + 30.0 * phase**4) / duration
    delta = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    return TrajectoryPoint(
        position=np.asarray(start, dtype=float) + blend * delta,
        velocity=blend_rate * delta,
    )
