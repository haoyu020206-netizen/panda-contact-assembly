from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

CONDITION_VARIANTS = {
    "mvp-clean": "mvp",
    "menagerie-clean": "menagerie",
    "menagerie-noisy": "menagerie",
    "menagerie-biased": "menagerie",
}
POLICY_KEYS = ("baseline", "learned", "pose-reference")
SUMMARY_FIELDS = (
    "episodes",
    "successes",
    "success_rate",
    "safety_violations",
    "mean_peak_force_n",
    "mean_reward",
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


def vector_is_zero(values: list[float]) -> bool:
    return len(values) == 3 and all(float(value) == 0.0 for value in values)


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
    if not summary_path.is_file():
        return [f"{prefix}: missing summary.json"]
    if not episodes_path.is_file():
        return [f"{prefix}: missing episodes.csv"]

    file_summary = load_json(summary_path)
    for field in SUMMARY_FIELDS:
        if field not in embedded_summary or field not in file_summary:
            failures.append(f"{prefix}: missing summary field {field}")
            continue
        if not math.isclose(
            float(embedded_summary[field]), float(file_summary[field]), rel_tol=0.0, abs_tol=1e-12
        ):
            failures.append(f"{prefix}: {field} disagrees with transfer_summary.json")

    try:
        episode_count = int(file_summary["episodes"])
        successes = int(file_summary["successes"])
        safety_violations = int(file_summary["safety_violations"])
        success_rate = float(file_summary["success_rate"])
    except (KeyError, TypeError, ValueError):
        failures.append(f"{prefix}: invalid summary counters")
        return failures
    if episode_count != expected_episodes:
        failures.append(f"{prefix}: episodes {episode_count} != expected {expected_episodes}")
    if not 0 <= successes <= episode_count:
        failures.append(f"{prefix}: successes outside valid range")
    if not 0 <= safety_violations <= episode_count:
        failures.append(f"{prefix}: safety violations outside valid range")
    expected_rate = successes / episode_count if episode_count else 0.0
    if not math.isclose(success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
        failures.append(f"{prefix}: success rate disagrees with successes/episodes")

    with episodes_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_episodes:
        failures.append(f"{prefix}: CSV rows {len(rows)} != expected {expected_episodes}")
        return failures
    try:
        csv_successes = sum(parse_bool(row["success"]) for row in rows)
        csv_safety = sum(parse_bool(row["safety_violation"]) for row in rows)
        csv_mean_force = math.fsum(float(row["max_contact_force_n"]) for row in rows) / len(rows)
        csv_mean_reward = math.fsum(float(row["total_reward"]) for row in rows) / len(rows)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"{prefix}: invalid episode CSV: {error}")
        return failures
    if csv_successes != successes:
        failures.append(f"{prefix}: CSV success count disagrees with summary")
    if csv_safety != safety_violations:
        failures.append(f"{prefix}: CSV safety count disagrees with summary")
    if not math.isclose(
        csv_mean_force, float(file_summary["mean_peak_force_n"]), rel_tol=0.0, abs_tol=1e-9
    ):
        failures.append(f"{prefix}: CSV mean force disagrees with summary")
    if not math.isclose(
        csv_mean_reward, float(file_summary["mean_reward"]), rel_tol=0.0, abs_tol=1e-9
    ):
        failures.append(f"{prefix}: CSV mean reward disagrees with summary")
    return failures


def validate_transfer(
    result_directory: Path,
    min_evaluation_episodes: int,
    min_target_learned_success_rate: float,
    max_target_learned_safety_violations: int | None,
    require_target_non_regression: bool,
) -> list[str]:
    summary = load_json(result_directory / "transfer_summary.json")
    failures: list[str] = []
    episode_count = int(summary["episodes"])
    if episode_count < min_evaluation_episodes:
        failures.append(f"evaluation episodes {episode_count} < required {min_evaluation_episodes}")

    conditions = summary.get("conditions", {})
    expected_keys = set(CONDITION_VARIANTS)
    actual_keys = set(conditions)
    if actual_keys != expected_keys:
        failures.append(f"condition keys {sorted(actual_keys)} != expected {sorted(expected_keys)}")
        return failures

    for condition_key, expected_variant in CONDITION_VARIANTS.items():
        condition = conditions[condition_key]
        if condition.get("model_variant") != expected_variant:
            failures.append(f"{condition_key}: unexpected model variant")
        policy_summaries = condition.get("policies", {})
        if set(policy_summaries) != set(POLICY_KEYS):
            failures.append(f"{condition_key}: unexpected policy keys")
            continue
        for policy_key in POLICY_KEYS:
            failures.extend(
                validate_policy_files(
                    result_directory,
                    condition_key,
                    policy_key,
                    policy_summaries[policy_key],
                    episode_count,
                )
            )

    clean_keys = ("mvp-clean", "menagerie-clean")
    for condition_key in clean_keys:
        condition = conditions[condition_key]
        if not vector_is_zero(condition["observation_position_noise_std_m"]):
            failures.append(f"{condition_key}: clean condition has observation noise")
        if not vector_is_zero(condition["observation_position_bias_m"]):
            failures.append(f"{condition_key}: clean condition has observation bias")
    noisy = conditions["menagerie-noisy"]
    noisy_std = noisy["observation_position_noise_std_m"]
    if len(noisy_std) != 3 or any(float(value) <= 0.0 for value in noisy_std):
        failures.append("menagerie-noisy: expected positive 3-D noise standard deviation")
    if not vector_is_zero(noisy["observation_position_bias_m"]):
        failures.append("menagerie-noisy: unexpected observation bias")
    biased = conditions["menagerie-biased"]
    if not vector_is_zero(biased["observation_position_noise_std_m"]):
        failures.append("menagerie-biased: unexpected observation noise")
    bias = biased["observation_position_bias_m"]
    if len(bias) != 3 or (float(bias[0]) == 0.0 and float(bias[1]) == 0.0) or float(bias[2]) != 0.0:
        failures.append("menagerie-biased: expected nonzero lateral 3-D bias")

    source_learned = conditions["mvp-clean"]["policies"]["learned"]
    target_learned = conditions["menagerie-clean"]["policies"]["learned"]
    expected_success_delta = float(target_learned["success_rate"]) - float(
        source_learned["success_rate"]
    )
    expected_force_delta = float(target_learned["mean_peak_force_n"]) - float(
        source_learned["mean_peak_force_n"]
    )
    if not math.isclose(
        float(summary["zero_shot_success_rate_delta"]),
        expected_success_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        failures.append("zero-shot success-rate delta disagrees with policy summaries")
    if not math.isclose(
        float(summary["zero_shot_peak_force_delta_n"]),
        expected_force_delta,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        failures.append("zero-shot peak-force delta disagrees with policy summaries")

    for condition_key in ("menagerie-clean", "menagerie-noisy", "menagerie-biased"):
        policies = conditions[condition_key]["policies"]
        baseline = policies["baseline"]
        learned = policies["learned"]
        learned_success = float(learned["success_rate"])
        learned_safety = int(learned["safety_violations"])
        if learned_success < min_target_learned_success_rate:
            failures.append(
                f"{condition_key}: learned success {learned_success:.3f} < required "
                f"{min_target_learned_success_rate:.3f}"
            )
        if (
            max_target_learned_safety_violations is not None
            and learned_safety > max_target_learned_safety_violations
        ):
            failures.append(
                f"{condition_key}: learned safety violations {learned_safety} > allowed "
                f"{max_target_learned_safety_violations}"
            )
        non_regression = learned_success >= float(
            baseline["success_rate"]
        ) and learned_safety <= int(baseline["safety_violations"])
        if require_target_non_regression and not non_regression:
            failures.append(f"{condition_key}: learned policy regressed against zero residual")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate residual-policy transfer evidence and acceptance thresholds."
    )
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--min-evaluation-episodes", type=int, default=1)
    parser.add_argument("--min-target-learned-success-rate", type=float, default=0.0)
    parser.add_argument("--max-target-learned-safety-violations", type=int)
    parser.add_argument("--require-target-non-regression", action="store_true")
    args = parser.parse_args()

    failures = validate_transfer(
        args.result_directory,
        min_evaluation_episodes=args.min_evaluation_episodes,
        min_target_learned_success_rate=args.min_target_learned_success_rate,
        max_target_learned_safety_violations=args.max_target_learned_safety_violations,
        require_target_non_regression=args.require_target_non_regression,
    )
    if failures:
        raise SystemExit("residual-transfer acceptance failed: " + "; ".join(failures))
    print("residual-transfer acceptance passed")


if __name__ == "__main__":
    main()
