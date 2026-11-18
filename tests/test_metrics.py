import numpy as np
import pytest

from panda_assembly.metrics import RunLog, minimum_geom_proximity


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


def test_minimum_geom_proximity_reports_nearest_point_axis_component(monkeypatch) -> None:
    candidates = {
        2: (0.010, np.array([0.0, 0.0, 0.0, 0.010, 0.0, 0.0])),
        3: (0.006, np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.006])),
    }

    def fake_geom_distance(model, data, source, target, cutoff, from_to):
        distance, points = candidates[target]
        from_to[:] = points
        return distance

    monkeypatch.setattr("panda_assembly.metrics.mujoco.mj_geomDistance", fake_geom_distance)
    distance, axis_component, lateral_gap = minimum_geom_proximity(
        object(), object(), 1, (2, 3), 0.020
    )

    assert distance == pytest.approx(0.006)
    assert axis_component == pytest.approx(1.0)
    assert lateral_gap == pytest.approx(0.0)
