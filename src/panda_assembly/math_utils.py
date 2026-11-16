from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def smoothstep(progress: float) -> float:
    """Cubic interpolation with zero endpoint velocity."""
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def rotation_error(current: FloatArray, desired: FloatArray) -> FloatArray:
    """World-frame orientation error from two 3x3 rotation matrices."""
    return 0.5 * sum(
        (np.cross(current[:, axis], desired[:, axis]) for axis in range(3)),
        start=np.zeros(3),
    )


def damped_task_nullspace(jacobian: FloatArray, damping: float) -> FloatArray:
    """Return a damped projector that keeps joint torques out of task space."""
    task_identity = np.eye(jacobian.shape[0])
    inverse = np.linalg.solve(
        jacobian @ jacobian.T + damping * damping * task_identity,
        task_identity,
    )
    return np.eye(jacobian.shape[1]) - jacobian.T @ inverse @ jacobian
