import numpy as np
import pytest

from panda_assembly.math_utils import (
    damped_task_nullspace,
    interpolate_rotation,
    rotation_error,
    smoothstep,
    vertical_tool_frame,
)


def test_smoothstep_clamps_and_has_expected_midpoint() -> None:
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(2.0) == 1.0


def test_rotation_error_is_zero_for_identical_frames() -> None:
    identity = np.eye(3)
    np.testing.assert_allclose(rotation_error(identity, identity), np.zeros(3))


def test_damped_nullspace_has_joint_shape() -> None:
    jacobian = np.zeros((6, 7))
    projector = damped_task_nullspace(jacobian, 0.01)
    np.testing.assert_allclose(projector, np.eye(7))


def test_vertical_tool_frame_preserves_rotation_matrix_and_points_z_up() -> None:
    tilted = np.array(
        [
            [0.8660254, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [-0.5, 0.0, 0.8660254],
        ]
    )
    vertical = vertical_tool_frame(tilted)
    np.testing.assert_allclose(vertical.T @ vertical, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(vertical[:, 2], [0.0, 0.0, 1.0])
    assert np.linalg.det(vertical) == pytest.approx(1.0)


def test_rotation_interpolation_hits_both_endpoints() -> None:
    start = np.eye(3)
    goal = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(interpolate_rotation(start, goal, 0.0), start)
    np.testing.assert_allclose(interpolate_rotation(start, goal, 1.0), goal, atol=1e-7)
