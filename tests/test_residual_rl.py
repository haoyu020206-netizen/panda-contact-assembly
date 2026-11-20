from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from panda_assembly.config import load_config
from panda_assembly.residual_rl import (
    ResidualActionFilter,
    ResidualPegInsertionEnv,
    collect_reference_dataset,
    pose_servo_reference_policy,
)

CONFIG = Path(__file__).parents[1] / "configs" / "mvp.toml"


def test_residual_action_filter_limits_rate_and_offset() -> None:
    action_filter = ResidualActionFilter(
        max_offset_m=np.array([0.008, 0.008, 0.003]),
        max_rate_m_s=np.array([0.020, 0.020, 0.010]),
    )
    np.testing.assert_allclose(action_filter.reset(), 0.0)
    first = action_filter.command(np.ones(3), 0.020, True)
    np.testing.assert_allclose(first, [0.0004, 0.0004, 0.0002])
    for _ in range(100):
        limited = action_filter.command(np.full(3, 2.0), 0.020, True)
    np.testing.assert_allclose(limited, [0.008, 0.008, 0.003])


def test_residual_environment_follows_gymnasium_contract() -> None:
    env = ResidualPegInsertionEnv(load_config(CONFIG))
    try:
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        assert not info["is_success"]
        next_observation, reward, terminated, truncated, next_info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert next_info["max_contact_force_n"] >= 0.0
    finally:
        env.close()


def test_residual_environment_rejects_out_of_shape_action() -> None:
    action_filter = ResidualActionFilter(np.ones(3), np.ones(3))
    with pytest.raises(ValueError, match="shape"):
        action_filter.command(np.ones(2), 0.02, True)


def test_pose_observation_perturbation_requires_valid_vector() -> None:
    vector = ResidualPegInsertionEnv._observation_vector(
        np.array([0.001, 0.002, 0.003]),
        "noise",
        nonnegative=True,
    )
    np.testing.assert_allclose(vector, [0.001, 0.002, 0.003])
    with pytest.raises(ValueError, match="shape"):
        ResidualPegInsertionEnv._observation_vector(np.ones(2), "noise")
    with pytest.raises(ValueError, match="nonnegative"):
        ResidualPegInsertionEnv._observation_vector(
            np.array([0.001, -0.001, 0.0]),
            "noise",
            nonnegative=True,
        )


def test_pose_servo_reference_maps_relative_error_to_bounded_offset() -> None:
    observation = np.zeros(12, dtype=np.float32)
    observation[:2] = [0.5, -1.0]
    action = pose_servo_reference_policy(observation)
    np.testing.assert_allclose(action, [0.75, -1.0, 0.0])


def test_action_and_observation_spaces_are_bounded() -> None:
    config = load_config(CONFIG)
    env = ResidualPegInsertionEnv(config)
    try:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert np.all(np.isfinite(env.action_space.low))
        assert np.all(np.isfinite(env.observation_space.high))
    finally:
        env.close()


def test_reference_dataset_has_aligned_observation_action_pairs() -> None:
    observations, actions = collect_reference_dataset(load_config(CONFIG), episodes=1, seed=5)
    assert observations.shape[0] == actions.shape[0]
    assert observations.shape[1:] == (12,)
    assert actions.shape[1:] == (3,)
    assert np.all(np.abs(actions) <= 1.0)
