from __future__ import annotations

import json

from scripts.aggregate_pose_estimator_transfer import (
    CONDITION_ORDER,
    POLICY_ORDER,
    aggregate_runs,
)
from scripts.validate_pose_estimator_transfer import validate_aggregate


def estimator_config(condition_key: str) -> dict:
    configs = {
        "oracle": (0.0, 0.0, [0.0, 0.0, 0.0], 0.0),
        "camera-nominal": (0.04, 0.0, [0.0005, 0.0005, 0.00025], 0.2),
        "camera-latency": (0.12, 0.0, [0.0005, 0.0005, 0.00025], 0.2),
        "camera-dropout": (0.04, 0.3, [0.0005, 0.0005, 0.00025], 0.2),
        "camera-correlated": (0.04, 0.0, [0.0015, 0.0015, 0.0005], 0.8),
        "camera-combined": (0.12, 0.3, [0.0015, 0.0015, 0.0005], 0.8),
    }
    latency, dropout, noise, correlation = configs[condition_key]
    return {
        "update_period_s": 0.02 if condition_key == "oracle" else 0.04,
        "latency_s": latency,
        "dropout_probability": dropout,
        "noise_std_m": noise,
        "bias_m": [0.0, 0.0, 0.0],
        "correlation_time_s": correlation,
    }


def policy_summary(condition_key: str, policy_key: str) -> dict:
    oracle = condition_key == "oracle"
    dropout = condition_key in {"camera-dropout", "camera-combined"}
    latency = estimator_config(condition_key)["latency_s"]
    successes = 2 if policy_key == "learned-search" else 1
    samples = 100
    dropped = 30 if dropout else 0
    return {
        "episodes": 2,
        "successes": successes,
        "success_rate": successes / 2,
        "safety_violations": 0,
        "mean_peak_force_n": 10.0,
        "mean_reward": 5.0,
        "contact_episodes": 1,
        "contact_rate": 0.5,
        "mean_contact_steps": 4.0,
        "search_triggers": 1 if policy_key == "learned-search" else 0,
        "recovery_successes": 1 if policy_key == "learned-search" else 0,
        "mean_pose_estimate_age_s": 0.0 if oracle else max(0.05, latency),
        "max_pose_estimate_age_s": 0.0 if oracle else max(0.08, latency + 0.04),
        "pose_estimator_samples": samples,
        "pose_estimator_dropouts": dropped,
        "pose_estimator_delivered_updates": samples - dropped,
        "pose_estimator_dropout_rate": dropped / samples,
    }


def run(seed: int) -> dict:
    return {
        "seed": seed,
        "episodes": 2,
        "socket_clearance_m": 0.003,
        "conditions": {
            condition_key: {
                "label": condition_key,
                "pose_estimator": estimator_config(condition_key),
                "policies": {
                    policy_key: policy_summary(condition_key, policy_key)
                    for policy_key in POLICY_ORDER
                },
            }
            for condition_key in CONDITION_ORDER
        },
    }


def test_pose_estimator_aggregate_and_acceptance(tmp_path) -> None:
    aggregate = aggregate_runs([run(11), run(12), run(13)])
    combined = aggregate["conditions"]["camera-combined"]["policies"]["learned-search"]
    assert combined["episodes"] == 6
    assert combined["success_rate"] == 1.0
    assert combined["pose_estimator_dropout_rate"] == 0.3

    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    failures = validate_aggregate(
        path,
        seed_results=None,
        min_seeds=3,
        min_evaluation_episodes=2,
        min_oracle_success_rate=0.5,
        min_combined_success_rate=0.5,
        max_combined_safety_violations=0,
        require_search_non_regression=True,
        require_estimator_signals=True,
    )
    assert failures == []


def test_pose_estimator_acceptance_detects_missing_dropout(tmp_path) -> None:
    aggregate = aggregate_runs([run(11), run(12), run(13)])
    aggregate["conditions"]["camera-dropout"]["policies"]["learned"][
        "pose_estimator_dropout_rate"
    ] = 0.0
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    failures = validate_aggregate(
        path,
        seed_results=None,
        min_seeds=3,
        min_evaluation_episodes=2,
        min_oracle_success_rate=0.0,
        min_combined_success_rate=0.0,
        max_combined_safety_violations=None,
        require_search_non_regression=False,
        require_estimator_signals=True,
    )
    assert any("camera-dropout did not record frame loss" in failure for failure in failures)
