from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CONDITION_ORDER = (
    "oracle",
    "camera-nominal",
    "camera-latency",
    "camera-dropout",
    "camera-correlated",
    "camera-combined",
)
POLICY_ORDER = ("learned", "learned-search")

SUMMARY_FIELDS = (
    "episodes",
    "successes",
    "success_rate",
    "safety_violations",
    "mean_peak_force_n",
    "mean_reward",
    "contact_episodes",
    "contact_rate",
    "mean_contact_steps",
    "search_triggers",
    "recovery_successes",
    "mean_pose_estimate_age_s",
    "max_pose_estimate_age_s",
    "pose_estimator_samples",
    "pose_estimator_dropouts",
    "pose_estimator_delivered_updates",
    "pose_estimator_dropout_rate",
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def validate_policy_files(
    run_directory: Path,
    condition_key: str,
    policy_key: str,
    embedded_summary: dict,
    expected_episodes: int,
) -> list[str]:
    prefix = f"seed-result/{condition_key}/{policy_key}"
    policy_directory = run_directory / condition_key / policy_key
    summary_path = policy_directory / "summary.json"
    episodes_path = policy_directory / "episodes.csv"
    if not summary_path.is_file() or not episodes_path.is_file():
        return [f"{prefix}: missing summary.json or episodes.csv"]
    file_summary = load_json(summary_path)
    failures: list[str] = []
    for field in SUMMARY_FIELDS:
        if field not in embedded_summary or field not in file_summary:
            failures.append(f"{prefix}: missing summary field {field}")
        elif not close(embedded_summary[field], file_summary[field]):
            failures.append(f"{prefix}: {field} disagrees with seed summary")
    if failures:
        return failures

    with episodes_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_episodes:
        return [f"{prefix}: CSV rows {len(rows)} != expected {expected_episodes}"]
    successes = sum(parse_bool(row["success"]) for row in rows)
    safety = sum(parse_bool(row["safety_violation"]) for row in rows)
    contacts = sum(parse_bool(row["contact_occurred"]) for row in rows)
    searches = sum(parse_bool(row["search_triggered"]) for row in rows)
    recoveries = sum(parse_bool(row["recovery_success"]) for row in rows)
    samples = sum(int(row["pose_estimator_samples"]) for row in rows)
    dropouts = sum(int(row["pose_estimator_dropouts"]) for row in rows)
    expected = {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "safety_violations": safety,
        "mean_peak_force_n": math.fsum(float(row["max_contact_force_n"]) for row in rows)
        / len(rows),
        "mean_reward": math.fsum(float(row["total_reward"]) for row in rows) / len(rows),
        "contact_episodes": contacts,
        "contact_rate": contacts / len(rows),
        "mean_contact_steps": math.fsum(float(row["contact_steps"]) for row in rows) / len(rows),
        "search_triggers": searches,
        "recovery_successes": recoveries,
        "mean_pose_estimate_age_s": math.fsum(
            float(row["mean_pose_estimate_age_s"]) for row in rows
        )
        / len(rows),
        "max_pose_estimate_age_s": max(float(row["max_pose_estimate_age_s"]) for row in rows),
        "pose_estimator_samples": samples,
        "pose_estimator_dropouts": dropouts,
        "pose_estimator_delivered_updates": sum(
            int(row["pose_estimator_delivered_updates"]) for row in rows
        ),
        "pose_estimator_dropout_rate": dropouts / samples if samples else 0.0,
    }
    for field, value in expected.items():
        if not close(file_summary[field], value):
            failures.append(f"{prefix}: CSV {field} disagrees with summary")
    if policy_key == "learned" and (searches or recoveries):
        failures.append(f"{prefix}: non-search policy recorded recovery metrics")
    return failures


def paired_outcomes(run_directory: Path, condition_key: str) -> dict[str, int]:
    rows = {}
    for policy_key in POLICY_ORDER:
        path = run_directory / condition_key / policy_key / "episodes.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows[policy_key] = list(csv.DictReader(handle))
    outcomes = {"rescued": 0, "harmed": 0, "both_success": 0, "neither": 0}
    for learned_row, search_row in zip(rows["learned"], rows["learned-search"], strict=True):
        learned_success = parse_bool(learned_row["success"])
        search_success = parse_bool(search_row["success"])
        if not learned_success and search_success:
            outcomes["rescued"] += 1
        elif learned_success and not search_success:
            outcomes["harmed"] += 1
        elif learned_success and search_success:
            outcomes["both_success"] += 1
        else:
            outcomes["neither"] += 1
    return outcomes


def validate_seed_results(
    root: Path,
    aggregate_seeds: set[int],
    aggregate_conditions: dict,
) -> list[str]:
    failures: list[str] = []
    found_seeds: set[int] = set()
    paired_totals = {
        condition_key: {"rescued": 0, "harmed": 0, "both_success": 0, "neither": 0}
        for condition_key in CONDITION_ORDER
    }
    for summary_path in sorted(root.rglob("pose_estimator_summary.json")):
        summary = load_json(summary_path)
        seed = int(summary["seed"])
        if seed in found_seeds:
            failures.append(f"duplicate seed result {seed}")
            continue
        found_seeds.add(seed)
        episodes = int(summary["episodes"])
        conditions = summary.get("conditions", {})
        if set(conditions) != set(CONDITION_ORDER):
            failures.append(f"seed {seed}: unexpected condition keys")
            continue
        for condition_key in CONDITION_ORDER:
            policies = conditions[condition_key].get("policies", {})
            if set(policies) != set(POLICY_ORDER):
                failures.append(f"seed {seed}/{condition_key}: unexpected policy keys")
                continue
            for policy_key in POLICY_ORDER:
                failures.extend(
                    validate_policy_files(
                        summary_path.parent,
                        condition_key,
                        policy_key,
                        policies[policy_key],
                        episodes,
                    )
                )
            try:
                seed_outcomes = paired_outcomes(summary_path.parent, condition_key)
            except (FileNotFoundError, ValueError) as error:
                failures.append(f"seed {seed}/{condition_key}: invalid paired episodes: {error}")
            else:
                for field, value in seed_outcomes.items():
                    paired_totals[condition_key][field] += value
    if found_seeds != aggregate_seeds:
        failures.append(
            f"seed result identities {sorted(found_seeds)} != aggregate {sorted(aggregate_seeds)}"
        )
    for condition_key, expected in paired_totals.items():
        actual = aggregate_conditions[condition_key].get("paired_outcomes")
        if actual != expected:
            failures.append(
                f"{condition_key}: aggregate paired outcomes disagree with episode CSVs"
            )
    return failures


def vector_nonzero(vector: list[float]) -> bool:
    return any(float(value) != 0.0 for value in vector)


def validate_estimator_signals(conditions: dict) -> list[str]:
    failures: list[str] = []
    oracle = conditions["oracle"]
    oracle_config = oracle["pose_estimator"]
    if (
        float(oracle_config["latency_s"]) != 0.0
        or float(oracle_config["dropout_probability"]) != 0.0
        or vector_nonzero(oracle_config["noise_std_m"])
    ):
        failures.append("oracle estimator must not contain latency, dropout, or noise")
    oracle_period_s = float(oracle_config["update_period_s"])
    for policy_key in POLICY_ORDER:
        policy = oracle["policies"][policy_key]
        if float(policy["max_pose_estimate_age_s"]) > oracle_period_s + 1e-9:
            failures.append(f"oracle/{policy_key}: estimate age exceeded its update period")
        if int(policy["pose_estimator_dropouts"]) != 0:
            failures.append(f"oracle/{policy_key}: unexpected dropped frames")

    nominal_config = conditions["camera-nominal"]["pose_estimator"]
    latency = conditions["camera-latency"]
    latency_s = float(latency["pose_estimator"]["latency_s"])
    if latency_s <= float(nominal_config["latency_s"]):
        failures.append("camera-latency must exceed nominal estimator latency")
    if float(latency["policies"]["learned"]["mean_pose_estimate_age_s"]) < 0.8 * latency_s:
        failures.append("camera-latency did not produce the configured estimate age")

    dropout = conditions["camera-dropout"]
    if float(dropout["pose_estimator"]["dropout_probability"]) <= 0.0:
        failures.append("camera-dropout must configure frame loss")
    if float(dropout["policies"]["learned"]["pose_estimator_dropout_rate"]) <= 0.1:
        failures.append("camera-dropout did not record frame loss")

    correlated = conditions["camera-correlated"]["pose_estimator"]
    if float(correlated["correlation_time_s"]) <= float(nominal_config["correlation_time_s"]):
        failures.append("camera-correlated must increase error correlation time")
    if max(float(value) for value in correlated["noise_std_m"]) <= max(
        float(value) for value in nominal_config["noise_std_m"]
    ):
        failures.append("camera-correlated must increase measurement error")

    combined = conditions["camera-combined"]
    combined_config = combined["pose_estimator"]
    if (
        float(combined_config["latency_s"]) <= 0.0
        or float(combined_config["dropout_probability"]) <= 0.0
    ):
        failures.append("camera-combined must configure both latency and dropout")
    if float(combined["policies"]["learned"]["pose_estimator_dropout_rate"]) <= 0.1:
        failures.append("camera-combined did not record frame loss")
    return failures


def validate_aggregate(
    aggregate_path: Path,
    *,
    seed_results: Path | None,
    min_seeds: int,
    min_evaluation_episodes: int,
    min_oracle_success_rate: float,
    min_combined_success_rate: float,
    min_all_learned_success_rate: float,
    max_learned_safety_violations_per_condition: int | None,
    max_combined_safety_violations: int | None,
    require_search_non_regression: bool,
    require_estimator_signals: bool,
) -> list[str]:
    aggregate = load_json(aggregate_path)
    failures: list[str] = []
    seeds = {int(seed) for seed in aggregate.get("seeds", [])}
    if len(seeds) < min_seeds:
        failures.append(f"seed count {len(seeds)} < required {min_seeds}")
    episode_counts = {int(value) for value in aggregate.get("episodes_per_seed", [])}
    if not episode_counts or min(episode_counts) < min_evaluation_episodes:
        failures.append("evaluation episodes per seed below required minimum")
    conditions = aggregate.get("conditions", {})
    if set(conditions) != set(CONDITION_ORDER):
        return failures + ["unexpected aggregate condition keys"]
    expected_episodes = sum(episode_counts) * len(seeds) if len(episode_counts) == 1 else None
    for condition_key in CONDITION_ORDER:
        policies = conditions[condition_key].get("policies", {})
        if set(policies) != set(POLICY_ORDER):
            failures.append(f"{condition_key}: unexpected policy keys")
            continue
        if expected_episodes is not None:
            for policy_key in POLICY_ORDER:
                if int(policies[policy_key]["episodes"]) != expected_episodes:
                    failures.append(f"{condition_key}/{policy_key}: aggregate episode mismatch")
        if require_search_non_regression:
            learned = policies["learned"]
            searched = policies["learned-search"]
            if float(searched["success_rate"]) < float(learned["success_rate"]):
                failures.append(f"{condition_key}: search regressed success")
            if int(searched["safety_violations"]) > int(learned["safety_violations"]):
                failures.append(f"{condition_key}: search regressed safety")
        learned = policies["learned"]
        if float(learned["success_rate"]) < min_all_learned_success_rate:
            failures.append(f"{condition_key}: learned success below required minimum")
        if (
            max_learned_safety_violations_per_condition is not None
            and int(learned["safety_violations"]) > max_learned_safety_violations_per_condition
        ):
            failures.append(f"{condition_key}: learned safety violations exceed limit")
    oracle_success = float(conditions["oracle"]["policies"]["learned"]["success_rate"])
    if oracle_success < min_oracle_success_rate:
        failures.append("oracle learned-policy success below required minimum")
    combined = conditions["camera-combined"]["policies"]["learned"]
    if float(combined["success_rate"]) < min_combined_success_rate:
        failures.append("combined-condition success below required minimum")
    if (
        max_combined_safety_violations is not None
        and int(combined["safety_violations"]) > max_combined_safety_violations
    ):
        failures.append("combined-condition safety violations exceed limit")
    if require_estimator_signals:
        failures.extend(validate_estimator_signals(conditions))
    if seed_results is not None:
        failures.extend(validate_seed_results(seed_results, seeds, conditions))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pose-estimator transfer evidence.")
    parser.add_argument("aggregate_json", type=Path)
    parser.add_argument("--seed-results", type=Path)
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--min-evaluation-episodes", type=int, default=1)
    parser.add_argument("--min-oracle-success-rate", type=float, default=0.0)
    parser.add_argument("--min-combined-success-rate", type=float, default=0.0)
    parser.add_argument("--min-all-learned-success-rate", type=float, default=0.0)
    parser.add_argument("--max-learned-safety-violations-per-condition", type=int)
    parser.add_argument("--max-combined-safety-violations", type=int)
    parser.add_argument("--require-search-non-regression", action="store_true")
    parser.add_argument("--require-estimator-signals", action="store_true")
    args = parser.parse_args()
    failures = validate_aggregate(
        args.aggregate_json,
        seed_results=args.seed_results,
        min_seeds=args.min_seeds,
        min_evaluation_episodes=args.min_evaluation_episodes,
        min_oracle_success_rate=args.min_oracle_success_rate,
        min_combined_success_rate=args.min_combined_success_rate,
        min_all_learned_success_rate=args.min_all_learned_success_rate,
        max_learned_safety_violations_per_condition=(
            args.max_learned_safety_violations_per_condition
        ),
        max_combined_safety_violations=args.max_combined_safety_violations,
        require_search_non_regression=args.require_search_non_regression,
        require_estimator_signals=args.require_estimator_signals,
    )
    if failures:
        raise SystemExit("pose-estimator acceptance failed: " + "; ".join(failures))
    print("pose-estimator acceptance passed")


if __name__ == "__main__":
    main()
