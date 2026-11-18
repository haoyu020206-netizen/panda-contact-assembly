import numpy as np
import pytest

from panda_assembly.safety import ForceSafetySupervisor, classify_peg_outcome


def test_force_supervisor_latches_and_generates_smooth_retraction() -> None:
    supervisor = ForceSafetySupervisor(
        force_limit_n=200.0,
        trigger_duration_s=0.0,
        hold_time_s=0.05,
        retract_distance_m=0.08,
        retract_duration_s=1.0,
    )
    position = np.array([0.4, 0.0, 0.3])
    rotation = np.eye(3)

    assert not supervisor.update(1.0, 199.9, position, rotation)
    assert supervisor.update(1.1, 200.0, position, rotation)
    assert supervisor.triggered
    assert not supervisor.update(1.2, 250.0, position, rotation)

    held, _, state = supervisor.command(1.12, position - 0.1, rotation)
    np.testing.assert_allclose(held, position)
    assert state == "stop"

    halfway, _, state = supervisor.command(1.65, position, rotation)
    assert state == "retract"
    assert halfway[2] == pytest.approx(position[2] + 0.04)

    retracted, _, state = supervisor.command(2.2, position, rotation)
    assert state == "retracted"
    assert retracted[2] == pytest.approx(position[2] + 0.08)


def test_force_supervisor_debounce_resets_below_limit() -> None:
    supervisor = ForceSafetySupervisor(200.0, 0.02, 0.05, 0.08, 1.0)
    position = np.zeros(3)
    rotation = np.eye(3)

    assert not supervisor.update(0.0, 210.0, position, rotation)
    assert not supervisor.update(0.01, 100.0, position, rotation)
    assert not supervisor.update(0.02, 210.0, position, rotation)
    assert supervisor.update(0.04, 210.0, position, rotation)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {"geometric_success": True, "max_contact_force_n": 100.0},
            ("success", None),
        ),
        (
            {"geometric_success": True, "max_contact_force_n": 300.0},
            ("unsafe_completion", "excessive_force"),
        ),
        (
            {
                "geometric_success": False,
                "max_contact_force_n": 250.0,
                "safety_triggered": True,
                "safety_recovered": True,
            },
            ("safety_abort_recovered", "excessive_force"),
        ),
        (
            {
                "geometric_success": False,
                "max_contact_force_n": 50.0,
                "final_contact_force_n": 10.0,
            },
            ("jammed", "jammed"),
        ),
        (
            {
                "geometric_success": False,
                "max_contact_force_n": 0.0,
                "final_xy_error_m": 0.01,
            },
            ("lateral_misalignment", "lateral_misalignment"),
        ),
    ],
)
def test_classify_peg_outcome(values: dict[str, object], expected: tuple[str, str | None]) -> None:
    defaults: dict[str, object] = {
        "geometric_success": False,
        "max_contact_force_n": 0.0,
        "final_contact_force_n": 0.0,
        "final_xy_error_m": 0.0,
        "final_insertion_depth_m": 0.0,
        "clearance_m": 0.005,
        "insertion_depth_m": 0.05,
        "force_limit_n": 200.0,
        "safety_triggered": False,
        "safety_recovered": False,
    }
    assert classify_peg_outcome(**(defaults | values)) == expected
