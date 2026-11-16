import numpy as np
import pytest

from panda_assembly.trajectories import quintic_joint_trajectory


def test_quintic_trajectory_hits_endpoints_with_zero_velocity() -> None:
    start = np.zeros(7)
    goal = np.ones(7)
    initial = quintic_joint_trajectory(start, goal, 0.0, 2.0)
    final = quintic_joint_trajectory(start, goal, 2.0, 2.0)
    np.testing.assert_allclose(initial.position, start)
    np.testing.assert_allclose(initial.velocity, np.zeros(7))
    np.testing.assert_allclose(final.position, goal)
    np.testing.assert_allclose(final.velocity, np.zeros(7), atol=1e-12)


def test_quintic_trajectory_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError):
        quintic_joint_trajectory(np.zeros(1), np.ones(1), 0.0, 0.0)
