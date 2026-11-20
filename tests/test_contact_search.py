from __future__ import annotations

import numpy as np
import pytest

from panda_assembly.contact_search import SpiralResidualPolicy


def zero_policy(observation: np.ndarray) -> np.ndarray:
    del observation
    return np.zeros(3, dtype=np.float32)


def observation(*, time_s: float, force_n: float = 0.0, relative_z_m: float = 0.02) -> np.ndarray:
    value = np.zeros(12, dtype=np.float32)
    value[2] = relative_z_m / 0.080
    value[6] = force_n / 200.0
    value[8] = time_s / 10.0
    return value


def make_policy() -> SpiralResidualPolicy:
    return SpiralResidualPolicy(
        zero_policy,
        max_position_offset_m=np.array([0.008, 0.008, 0.003]),
        episode_duration_s=10.0,
        force_scale_n=200.0,
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
