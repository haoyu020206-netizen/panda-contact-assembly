from __future__ import annotations

import numpy as np
import pytest

from panda_assembly.pose_estimation import DelayedPoseEstimator, PoseEstimatorConfig


def test_estimator_delays_and_holds_measurements() -> None:
    estimator = DelayedPoseEstimator(PoseEstimatorConfig(update_period_s=0.02, latency_s=0.04))
    rng = np.random.default_rng(7)
    initial = estimator.reset(np.zeros(3), 0.0, rng)
    assert initial.age_s == 0.0

    before_delivery = estimator.observe(np.ones(3), 0.02, rng)
    np.testing.assert_allclose(before_delivery.relative_position_m, 0.0)
    assert before_delivery.age_s == pytest.approx(0.02)
    still_held = estimator.observe(np.full(3, 2.0), 0.04, rng)
    np.testing.assert_allclose(still_held.relative_position_m, 0.0)
    delivered = estimator.observe(np.full(3, 3.0), 0.06, rng)
    np.testing.assert_allclose(delivered.relative_position_m, 1.0)
    assert delivered.measurement_time_s == pytest.approx(0.02)
    assert delivered.age_s == pytest.approx(0.04)


def test_estimator_dropout_holds_warm_start() -> None:
    config = PoseEstimatorConfig(update_period_s=0.02, dropout_probability=1.0)
    estimator = DelayedPoseEstimator(config)
    rng = np.random.default_rng(3)
    estimator.reset(np.array([0.1, 0.2, 0.3]), 0.0, rng)
    estimate = estimator.observe(np.ones(3), 0.02, rng)
    np.testing.assert_allclose(estimate.relative_position_m, [0.1, 0.2, 0.3])
    assert estimate.samples == 2
    assert estimate.dropouts == 1
    assert estimate.delivered_updates == 1


def test_estimator_correlated_error_is_reproducible() -> None:
    config = PoseEstimatorConfig(
        update_period_s=0.02,
        noise_std_m=(0.001, 0.001, 0.001),
        correlation_time_s=0.5,
    )
    first = DelayedPoseEstimator(config)
    second = DelayedPoseEstimator(config)
    first_rng = np.random.default_rng(11)
    second_rng = np.random.default_rng(11)
    first.reset(np.zeros(3), 0.0, first_rng)
    second.reset(np.zeros(3), 0.0, second_rng)
    for step in range(1, 6):
        time_s = step * 0.02
        first_estimate = first.observe(np.zeros(3), time_s, first_rng)
        second_estimate = second.observe(np.zeros(3), time_s, second_rng)
        np.testing.assert_allclose(
            first_estimate.relative_position_m,
            second_estimate.relative_position_m,
        )


def test_pose_estimator_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        PoseEstimatorConfig(update_period_s=0.0)
    with pytest.raises(ValueError, match="dropout_probability"):
        PoseEstimatorConfig(dropout_probability=1.1)
    with pytest.raises(ValueError, match="nonnegative"):
        PoseEstimatorConfig(noise_std_m=(0.0, -0.001, 0.0))
