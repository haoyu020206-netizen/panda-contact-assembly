from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CONDITION_KEYS = ("wide-clean", "narrow-clean", "narrow-biased")
POLICY_KEYS = ("baseline", "learned", "learned-search", "pose-reference")
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


def validate_policy_files(
    result_directory: Path,
    condition_key: str,
    policy_key: str,
    embedded_summary: dict,
    expected_episodes: int,
) -> list[str]:
    prefix = f"{condition_key}/{policy_key}"
    failures: list[str] = []
    policy_directory = result_directory / condition_key / policy_key
    summary_path = policy_directory / "summary.json"
    episodes_path = policy_directory / "episodes.csv"
    if not summary_path.is_file() or not episodes_path.is_file():
        return [f"{prefix}: missing summary.json or episodes.csv"]
    file_summary = load_json(summary_path)
    for field in SUMMARY_FIELDS:
        if field not in embedded_summary or field not in file_summary:
            failures.append(f"{prefix}: missing summary field {field}")
            continue
        if not math.isclose(
            float(embedded_summary[field]), float(file_summary[field]), rel_tol=0.0, abs_tol=1e-9
        ):
            failures.append(f"{prefix}: {field} disagrees with contact_rich_summary.json")
    if int(file_summary.get("episodes", -1)) != expected_episodes:
        actual_episodes = file_summary.get("episodes")
        failures.append(
            f"{prefix}: summary episodes {actual_episodes} != expected {expected_episodes}"
        )
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
    mean_force = math.fsum(float(row["max_contact_force_n"]) for row in rows) / len(rows)
    mean_reward = math.fsum(float(row["total_reward"]) for row in rows) / len(rows)
    mean_contact_steps = math.fsum(float(row["contact_steps"]) for row in rows) / len(rows)
    expected_values = {
        "successes": successes,
        "safety_violations": safety,
        "contact_episodes": contacts,
        "search_triggers": searches,
        "recovery_successes": recoveries,
        "mean_peak_force_n": mean_force,
        "mean_reward": mean_reward,
        "mean_contact_steps": mean_contact_steps,
    }
    for field, expected in expected_values.items():
        if not math.isclose(float(file_summary[field]), float(expected), rel_tol=0.0, abs_tol=1e-9):
            failures.append(f"{prefix}: CSV {field} disagrees with summary")
    if not math.isclose(
        float(file_summary["success_rate"]), successes / len(rows), rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append(f"{prefix}: success rate disagrees with CSV")
    if not math.isclose(
        float(file_summary["contact_rate"]), contacts / len(rows), rel_tol=0.0, abs_tol=1e-12
    ):
        failures.append(f"{prefix}: contact rate disagrees with CSV")
    if policy_key != "learned-search" and (searches or recoveries):
        failures.append(f"{prefix}: non-search policy recorded search metrics")
    return failures


def paired_outcomes(result_directory: Path, condition_key: str) -> dict[str, int]:
    def rows(policy_key: str) -> list[dict[str, str]]:
        path = result_directory / condition_key / policy_key / "episodes.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    learned = rows("learned")
    searched = rows("learned-search")
    outcomes = {"rescued": 0, "harmed": 0, "both_success": 0, "neither": 0}
    for learned_row, search_row in zip(learned, searched, strict=True):
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


def validate_contact_rich(
    result_directory: Path,
    *,
    min_evaluation_episodes: int,
    min_narrow_contact_rate: float,
    max_search_safety_violations: int | None,
    min_total_rescued: int,
    require_search_non_regression: bool,
) -> list[str]:
    summary = load_json(result_directory / "contact_rich_summary.json")
    failures: list[str] = []
    episode_count = int(summary["episodes"])
    if episode_count < min_evaluation_episodes:
        failures.append(f"evaluation episodes {episode_count} < required {min_evaluation_episodes}")
    conditions = summary.get("conditions", {})
    if set(conditions) != set(CONDITION_KEYS):
        return failures + ["unexpected contact-rich condition keys"]

    wide_clearance = float(conditions["wide-clean"]["socket_clearance_m"])
    narrow_clearance = float(conditions["narrow-clean"]["socket_clearance_m"])
    biased_clearance = float(conditions["narrow-biased"]["socket_clearance_m"])
    if not 0.0 < narrow_clearance < wide_clearance or biased_clearance != narrow_clearance:
        failures.append(
            "narrow conditions must share a positive clearance below the wide condition"
        )
    if any(
        float(value) != 0.0 for value in conditions["narrow-clean"]["observation_position_bias_m"]
    ):
        failures.append("narrow-clean must not contain observation bias")
    bias = conditions["narrow-biased"]["observation_position_bias_m"]
    if len(bias) != 3 or (float(bias[0]) == 0.0 and float(bias[1]) == 0.0):
        failures.append("narrow-biased must contain nonzero lateral observation bias")

    total_rescued = 0
    for condition_key in CONDITION_KEYS:
        policies = conditions[condition_key].get("policies", {})
        if set(policies) != set(POLICY_KEYS):
            failures.append(f"{condition_key}: unexpected policy keys")
            continue
        for policy_key in POLICY_KEYS:
            failures.extend(
                validate_policy_files(
                    result_directory,
                    condition_key,
                    policy_key,
                    policies[policy_key],
                    episode_count,
                )
            )
        if condition_key == "wide-clean":
            continue
        learned = policies["learned"]
        searched = policies["learned-search"]
        if float(learned["contact_rate"]) < min_narrow_contact_rate:
            failures.append(f"{condition_key}: learned contact rate below benchmark minimum")
        if (
            max_search_safety_violations is not None
            and int(searched["safety_violations"]) > max_search_safety_violations
        ):
            failures.append(f"{condition_key}: search policy exceeded safety limit")
        if require_search_non_regression:
            if float(searched["success_rate"]) < float(learned["success_rate"]):
                failures.append(f"{condition_key}: search policy regressed task success")
            if int(searched["safety_violations"]) > int(learned["safety_violations"]):
                failures.append(f"{condition_key}: search policy regressed safety")
            if float(searched["mean_peak_force_n"]) > float(learned["mean_peak_force_n"]) + 1e-3:
                failures.append(f"{condition_key}: search policy increased mean peak force")
        outcomes = paired_outcomes(result_directory, condition_key)
        if outcomes["harmed"] > outcomes["rescued"]:
            failures.append(f"{condition_key}: paired search outcomes have net harm")
        total_rescued += outcomes["rescued"]
    if total_rescued < min_total_rescued:
        failures.append(f"paired rescues {total_rescued} < required {min_total_rescued}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate contact-rich transfer evidence.")
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--min-evaluation-episodes", type=int, default=1)
    parser.add_argument("--min-narrow-contact-rate", type=float, default=0.0)
    parser.add_argument("--max-search-safety-violations", type=int)
    parser.add_argument("--min-total-rescued", type=int, default=0)
    parser.add_argument("--require-search-non-regression", action="store_true")
    args = parser.parse_args()
    failures = validate_contact_rich(
        args.result_directory,
        min_evaluation_episodes=args.min_evaluation_episodes,
        min_narrow_contact_rate=args.min_narrow_contact_rate,
        max_search_safety_violations=args.max_search_safety_violations,
        min_total_rescued=args.min_total_rescued,
        require_search_non_regression=args.require_search_non_regression,
    )
    if failures:
        raise SystemExit("contact-rich acceptance failed: " + "; ".join(failures))
    print("contact-rich acceptance passed")


if __name__ == "__main__":
    main()
