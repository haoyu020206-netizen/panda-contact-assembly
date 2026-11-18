import numpy as np
import pytest

from panda_assembly.safety import (
    ForceSafetySupervisor,
    InsertionSpeedLimiter,
    SpiralRecoverySupervisor,
    classify_peg_outcome,
)


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


def test_insertion_speed_limiter_slows_near_surface_and_on_contact() -> None:
    limiter = InsertionSpeedLimiter(
        timestep_s=0.01,
        precontact_distance_m=0.015,
        slow_zone_depth_m=0.010,
        free_speed_m_s=0.020,
        precontact_speed_m_s=0.005,
        contact_slow_force_n=20.0,
        contact_speed_m_s=0.001,
    )
    nominal = np.array([0.4, 0.0, 0.2])
    limiter.command(np.array([0.4, 0.0, 0.3]), 0.35, 0.30, 0.0, False)

    slowed, state = limiter.command(nominal, 0.31, 0.30, 0.0, True)
    assert state == "precontact_slow"
    assert slowed[2] == pytest.approx(0.29995)

    contact_slowed, state = limiter.command(nominal, 0.31, 0.30, 25.0, True)
    assert state == "contact_slow"
    assert contact_slowed[2] == pytest.approx(0.29994)


def test_spiral_recovery_detects_jam_searches_and_retries() -> None:
    position = np.array([0.4, 0.0, 0.3])
    rotation = np.eye(3)
    supervisor = SpiralRecoverySupervisor(
        jam_force_n=5.0,
        jam_timeout_s=0.6,
        jam_progress_m=0.001,
        retract_distance_m=0.015,
        retract_duration_s=0.6,
        spiral_radius_m=0.008,
        spiral_cycles=4,
        spiral_duration_s=4.0,
        spiral_descent_m=0.012,
        entry_depth_m=0.004,
        retry_insert_duration_s=2.0,
        max_retries=1,
        search_center=position.copy(),
        final_target_z_m=0.24,
    )

    assert not supervisor.update(0.0, 8.0, 0.0, False, position, rotation)
    assert supervisor.update(0.61, 8.0, 0.0, False, position, rotation)
    assert supervisor.state_name == "local_retract"

    target, _, state = supervisor.command(0.91, position, rotation)
    assert state == "local_retract"
    assert target[2] == pytest.approx(0.3075)

    supervisor.update(1.21, 0.0, 0.0, False, position, rotation)
    assert supervisor.state_name == "spiral_search"
    target, _, state = supervisor.command(4.21, position, rotation)
    assert state == "spiral_search"
    assert target[0] == pytest.approx(position[0] + 0.006, abs=1e-9)
    assert target[1] == pytest.approx(position[1], abs=1e-9)
    assert target[2] < position[2]

    entry_position = np.array([0.406, 0.0, 0.292])
    supervisor.update(4.21, 3.0, 0.005, False, entry_position, rotation)
    assert supervisor.entry_detected
    assert supervisor.state_name == "retry_insert"
    target, _, state = supervisor.command(5.21, position, rotation)
    assert state == "retry_insert"
    assert target[0] == pytest.approx(0.406)
    assert target[2] == pytest.approx((0.292 + 0.24) / 2.0)

    supervisor.update(5.3, 0.0, 0.05, True, entry_position, rotation)
    assert supervisor.retry_success
    assert supervisor.state_name == "completed"


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
        (
            {
                "geometric_success": False,
                "contact_recovery_triggered": True,
            },
            ("recovery_failed", "jammed"),
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
