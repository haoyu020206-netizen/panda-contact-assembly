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


def vertical_tool_frame(current: FloatArray) -> FloatArray:
    """Build the closest yaw-preserving frame whose local z-axis points upward."""
    world_z = np.array([0.0, 0.0, 1.0])
    x_axis = current[:, 0].copy()
    x_axis[2] = 0.0
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.cross(current[:, 1], world_z)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(world_z, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, world_z)
    return np.column_stack((x_axis, y_axis, world_z))


def interpolate_rotation(start: FloatArray, goal: FloatArray, progress: float) -> FloatArray:
    """Geodesically interpolate two rotation matrices using an SO(3) exponential."""
    amount = smoothstep(progress)
    relative = start.T @ goal
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return start.copy()
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) / (2.0 * np.sin(angle))
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    step_angle = amount * angle
    step = np.eye(3) + np.sin(step_angle) * skew + (1.0 - np.cos(step_angle)) * (skew @ skew)
    return start @ step
