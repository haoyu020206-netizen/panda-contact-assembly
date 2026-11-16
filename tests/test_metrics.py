import numpy as np
import pytest

from panda_assembly.metrics import RunLog


def test_position_and_joint_errors_are_kept_in_their_own_units() -> None:
    cartesian_log = RunLog()
    cartesian_log.append(
        0.0,
        np.array([0.003, 0.004, 0.0]),
        2.0,
        np.zeros(7),
        orientation_error=np.array([0.0, 0.01, 0.0]),
    )
    np.testing.assert_allclose(cartesian_log.position_errors(), [0.005])
    np.testing.assert_allclose(cartesian_log.orientation_errors(), [0.01])
    assert cartesian_log.joint_errors().size == 0

    joint_log = RunLog()
    joint_log.append(
        0.0,
        None,
        0.0,
        np.zeros(7),
        joint_error=np.full(7, 0.1),
    )
    assert joint_log.position_errors().size == 0
    assert joint_log.orientation_errors().size == 0
    assert joint_log.joint_errors()[0] == pytest.approx(np.sqrt(0.07))
