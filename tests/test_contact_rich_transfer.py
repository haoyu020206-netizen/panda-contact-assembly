from __future__ import annotations

import csv
import json

from scripts.validate_contact_rich_transfer import (
    CONDITION_KEYS,
    POLICY_KEYS,
    paired_outcomes,
    validate_contact_rich,
)


def write_policy(directory, outcomes: list[tuple[bool, bool, bool]]) -> dict:
    directory.mkdir(parents=True)
    rows = []
    for episode, (success, contact, search) in enumerate(outcomes):
        rows.append(
            {
                "episode": episode,
                "success": success,
                "safety_violation": False,
                "total_reward": 10.0 + episode,
                "max_contact_force_n": 5.0 + episode,
                "contact_occurred": contact,
                "contact_steps": 2 + episode if contact else 0,
                "search_triggered": search,
                "recovery_success": search and success,
            }
        )
    episodes = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    contacts = sum(bool(row["contact_occurred"]) for row in rows)
    summary = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "safety_violations": 0,
        "mean_peak_force_n": sum(float(row["max_contact_force_n"]) for row in rows) / episodes,
        "mean_reward": sum(float(row["total_reward"]) for row in rows) / episodes,
        "contact_episodes": contacts,
        "contact_rate": contacts / episodes,
        "mean_contact_steps": sum(float(row["contact_steps"]) for row in rows) / episodes,
        "search_triggers": sum(bool(row["search_triggered"]) for row in rows),
        "recovery_successes": sum(bool(row["recovery_success"]) for row in rows),
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (directory / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def write_result(root) -> None:
    conditions = {}
    for condition_key in CONDITION_KEYS:
        narrow = condition_key != "wide-clean"
        policy_summaries = {}
        for policy_key in POLICY_KEYS:
            if policy_key == "baseline":
                outcomes = [(False, narrow, False), (False, narrow, False)]
            elif policy_key == "learned":
                outcomes = [(True, narrow, False), (False, narrow, False)]
            elif policy_key == "learned-search":
                outcomes = [(True, narrow, False), (True, narrow, narrow)]
            else:
                outcomes = [(True, narrow, False), (True, narrow, False)]
            policy_summaries[policy_key] = write_policy(root / condition_key / policy_key, outcomes)
        conditions[condition_key] = {
            "label": condition_key,
            "model_variant": "menagerie",
            "socket_clearance_m": 0.003 if narrow else 0.005,
            "observation_position_bias_m": (
                [0.004, -0.004, 0.0] if condition_key == "narrow-biased" else [0.0] * 3
            ),
            "policies": policy_summaries,
        }
    summary = {"episodes": 2, "conditions": conditions}
    (root / "contact_rich_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_contact_rich_validator_accepts_paired_rescues(tmp_path) -> None:
    write_result(tmp_path)
    failures = validate_contact_rich(
        tmp_path,
        min_evaluation_episodes=2,
        min_narrow_contact_rate=1.0,
        max_search_safety_violations=0,
        min_total_rescued=2,
        require_search_non_regression=True,
    )
    assert failures == []
    assert paired_outcomes(tmp_path, "narrow-clean") == {
        "rescued": 1,
        "harmed": 0,
        "both_success": 1,
        "neither": 0,
    }


def test_contact_rich_validator_detects_csv_disagreement(tmp_path) -> None:
    write_result(tmp_path)
    path = tmp_path / "narrow-clean" / "learned" / "episodes.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("True,False", "False,False", 1))
    failures = validate_contact_rich(
        tmp_path,
        min_evaluation_episodes=1,
        min_narrow_contact_rate=0.0,
        max_search_safety_violations=None,
        min_total_rescued=0,
        require_search_non_regression=False,
    )
    assert any("CSV successes disagrees" in failure for failure in failures)


def test_contact_rich_validator_detects_summary_episode_disagreement(tmp_path) -> None:
    write_result(tmp_path)
    policy_summary = tmp_path / "narrow-clean" / "learned" / "summary.json"
    summary = json.loads(policy_summary.read_text(encoding="utf-8"))
    summary["episodes"] = 3
    policy_summary.write_text(json.dumps(summary), encoding="utf-8")
    aggregate = tmp_path / "contact_rich_summary.json"
    aggregate_summary = json.loads(aggregate.read_text(encoding="utf-8"))
    aggregate_summary["conditions"]["narrow-clean"]["policies"]["learned"]["episodes"] = 3
    aggregate.write_text(json.dumps(aggregate_summary), encoding="utf-8")

    failures = validate_contact_rich(
        tmp_path,
        min_evaluation_episodes=1,
        min_narrow_contact_rate=0.0,
        max_search_safety_violations=None,
        min_total_rescued=0,
        require_search_non_regression=False,
    )
    assert any("summary episodes 3 != expected 2" in failure for failure in failures)
