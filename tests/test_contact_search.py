from __future__ import annotations

import numpy as np
import pytest

from panda_assembly.contact_search import EntryMode, SpiralResidualPolicy, StallTriggerConfig
from panda_assembly.pose_estimation import PoseEstimate


def zero_policy(observation: np.ndarray) -> np.ndarray:
    del observation
    return np.zeros(3, dtype=np.float32)


def observation(*, time_s: float, force_n: float = 0.0, relative_z_m: float = 0.02) -> np.ndarray:
    value = np.zeros(12, dtype=np.float32)
    value[2] = relative_z_m / 0.080
    value[6] = force_n / 200.0
    value[8] = time_s / 10.0
    return value


def make_policy(
    entry_mode: EntryMode = "legacy", stall_trigger: StallTriggerConfig | None = None
) -> SpiralResidualPolicy:
    return SpiralResidualPolicy(
        zero_policy,
        max_position_offset_m=np.array([0.008, 0.008, 0.003]),
        episode_duration_s=10.0,
        force_scale_n=200.0,
        entry_mode=entry_mode,
        stall_trigger=stall_trigger,
    )


def test_spiral_policy_unloads_searches_and_holds_entry_action() -> None:
    policy = make_policy()

    assert np.allclose(policy(observation(time_s=4.0)), 0.0)
    unload = policy(observation(time_s=4.1, force_n=10.0))
    assert unload[2] == pytest.approx(1.0)
    assert policy.episode_metrics() == {"search_triggered": True, "entry_detected": False}

    search = policy(observation(time_s=6.0, force_n=3.0))
    assert np.linalg.norm(search[:2]) > 0.0
    entry = policy(observation(time_s=6.1, relative_z_m=-0.004))
    assert np.allclose(entry, search)
    assert policy.episode_metrics() == {"search_triggered": True, "entry_detected": True}

    policy.reset_episode()
    assert policy.episode_metrics() == {"search_triggered": False, "entry_detected": False}


def test_spiral_policy_rejects_radius_beyond_action_limit() -> None:
    with pytest.raises(ValueError, match="radius"):
        SpiralResidualPolicy(
            zero_policy,
            max_position_offset_m=np.array([0.002, 0.002, 0.003]),
            episode_duration_s=10.0,
            force_scale_n=200.0,
            search_radius_m=0.004,
        )


def test_recovery_uses_positive_depth_after_retraction() -> None:
    policy = make_policy("depth")
    assert policy(observation(time_s=0.0, force_n=10.0, relative_z_m=0.005))[2] == 0.0
    assert policy(observation(time_s=1.0, force_n=10.0, relative_z_m=-0.005))[2] == 1.0
    policy(observation(time_s=1.25, relative_z_m=0.005))
    assert not policy.episode_metrics()["entry_detected"]
    policy(observation(time_s=1.6, relative_z_m=-0.005))
    assert not policy.episode_metrics()["entry_detected"]
    policy(observation(time_s=1.7, relative_z_m=0.005))
    assert policy.episode_metrics()["entry_detected"]


def test_compensated_recovery_rejects_old_frames_and_accounts_for_tip_motion() -> None:
    policy = make_policy("compensated")

    def step(time_s, tip_z_m, capture_s, depth_m, force_n=0.0):
        pose = PoseEstimate(
            np.array([0.0, 0.0, depth_m]), capture_s, time_s - capture_s, True, 1, 0, 1
        )
        return policy.act_with_pose(
            observation(time_s=time_s, force_n=force_n, relative_z_m=depth_m),
            pose,
            time_s=time_s,
            tip_z_m=tip_z_m,
        )

    assert step(0.2, 0.01, 0.0, -0.01, 10.0)[2] == 0.0
    assert step(1.0, 0.01, 1.0, -0.01, 10.0)[2] == 1.0
    step(1.6, 0.001, 1.4, 0.005)
    assert not policy.episode_metrics()["entry_detected"]
    step(1.64, 0.001, 1.64, -0.001)
    step(1.76, -0.005, 1.64, -0.001)
    assert policy.episode_metrics()["entry_detected"]
    policy.reset_episode()
    assert step(1.76, -0.005, 1.64, -0.001, 10.0)[2] == 0.0


def test_stall_trigger_waits_for_contact_and_resets_on_progress_or_release() -> None:
    policy = make_policy("compensated", StallTriggerConfig(0.6, 0.001))

    def step(time_s: float, tip_z_m: float, force_n: float = 10.0) -> float:
        pose = PoseEstimate(np.array([0.0, 0.0, 0.0]), time_s, 0.0, True, 1, 0, 1)
        action = policy.act_with_pose(
            observation(time_s=time_s, force_n=force_n), pose, time_s=time_s, tip_z_m=tip_z_m
        )
        return float(action[2])

    assert step(1.0, 0.010) == 0.0
    assert step(1.4, 0.008) == 0.0  # Advancing resets the window.
    assert step(1.7, 0.008, force_n=0.0) == 0.0
    assert step(2.0, 0.008) == 0.0
    assert step(2.58, 0.008) == 0.0
    assert step(2.6, 0.008) == 1.0
    assert policy.episode_metrics()["search_triggered"]


def test_stall_trigger_requires_fresh_pose_and_time_for_recovery() -> None:
    policy = make_policy("compensated", StallTriggerConfig(0.6, 0.001))

    def step(time_s: float, age_s: float) -> None:
        pose = PoseEstimate(np.zeros(3), time_s - age_s, age_s, age_s == 0.0, 1, 0, 1)
        policy.act_with_pose(
            observation(time_s=time_s, force_n=10.0), pose, time_s=time_s, tip_z_m=0.0
        )

    step(1.0, 0.0)
    step(1.6, 0.25)
    assert policy.trigger_decision == "stale_pose"
    step(1.7, 0.0)
    assert policy.episode_metrics()["search_triggered"]
    policy.reset_episode()
    step(4.6, 0.0)
    step(5.2, 0.0)
    assert policy.trigger_decision == "insufficient_time"
    assert not policy.episode_metrics()["search_triggered"]
