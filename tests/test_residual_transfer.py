from __future__ import annotations

import csv
import json

from scripts.validate_residual_transfer import CONDITION_VARIANTS, POLICY_KEYS, validate_transfer


def write_policy(directory, successes: int, safety: int, episodes: int = 2) -> dict:
    directory.mkdir(parents=True)
    rows = []
    for episode in range(episodes):
        rows.append(
            {
                "episode": episode,
                "success": episode < successes,
                "safety_violation": episode < safety,
                "total_reward": 10.0 + episode,
                "max_contact_force_n": 2.0 + episode,
            }
        )
    summary = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "safety_violations": safety,
        "mean_peak_force_n": sum(float(row["max_contact_force_n"]) for row in rows) / episodes,
        "mean_reward": sum(float(row["total_reward"]) for row in rows) / episodes,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def write_transfer_result(root) -> None:
    conditions = {}
    for condition_key, variant in CONDITION_VARIANTS.items():
        policies = {}
        for policy_key in POLICY_KEYS:
            successes = 0 if policy_key == "baseline" else 2
            safety = 1 if policy_key == "baseline" else 0
            policies[policy_key] = write_policy(
                root / condition_key / policy_key, successes=successes, safety=safety
            )
        noise = [0.001, 0.001, 0.001] if condition_key == "menagerie-noisy" else [0.0] * 3
        bias = [0.002, -0.002, 0.0] if condition_key == "menagerie-biased" else [0.0] * 3
        conditions[condition_key] = {
            "label": condition_key,
            "model_variant": variant,
            "observation_position_noise_std_m": noise,
            "observation_position_bias_m": bias,
            "policies": policies,
        }
    source = conditions["mvp-clean"]["policies"]["learned"]
    target = conditions["menagerie-clean"]["policies"]["learned"]
    summary = {
        "seed": 1,
        "episodes": 2,
        "conditions": conditions,
        "zero_shot_success_rate_delta": target["success_rate"] - source["success_rate"],
        "zero_shot_peak_force_delta_n": target["mean_peak_force_n"] - source["mean_peak_force_n"],
    }
    (root / "transfer_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_validate_transfer_accepts_consistent_non_regression(tmp_path) -> None:
    write_transfer_result(tmp_path)

    failures = validate_transfer(
        tmp_path,
        min_evaluation_episodes=2,
        min_target_learned_success_rate=0.9,
        max_target_learned_safety_violations=0,
        require_target_non_regression=True,
    )

    assert failures == []


def test_validate_transfer_detects_csv_disagreement(tmp_path) -> None:
    write_transfer_result(tmp_path)
    episodes_path = tmp_path / "menagerie-clean" / "learned" / "episodes.csv"
    contents = episodes_path.read_text(encoding="utf-8")
    episodes_path.write_text(contents.replace("True,False", "False,False", 1), encoding="utf-8")

    failures = validate_transfer(
        tmp_path,
        min_evaluation_episodes=2,
        min_target_learned_success_rate=0.0,
        max_target_learned_safety_violations=None,
        require_target_non_regression=False,
    )

    assert "menagerie-clean/learned: CSV success count disagrees with summary" in failures
