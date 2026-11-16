import numpy as np

from panda_assembly.math_utils import damped_task_nullspace, rotation_error, smoothstep


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
