from __future__ import annotations

import argparse
import json
from pathlib import Path

POLICY_KEYS = {
    "baseline": ("baseline_success_rate", "baseline_safety_violations"),
    "behavior-cloned": (
        "behavior_cloned_success_rate",
        "behavior_cloned_safety_violations",
    ),
    "learned": ("learned_success_rate", "learned_safety_violations"),
    "pose-reference": (
        "pose_reference_success_rate",
        "pose_reference_safety_violations",
    ),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_results(
    result_directory: Path,
    min_evaluation_episodes: int,
    min_learned_success_rate: float,
    max_learned_safety_violations: int | None,
    require_non_regression: bool,
) -> list[str]:
    comparison = load_json(result_directory / "comparison.json")
    failures: list[str] = []
    episode_count = int(comparison["evaluation_episodes"])
    if episode_count < min_evaluation_episodes:
        failures.append(f"evaluation episodes {episode_count} < required {min_evaluation_episodes}")

    for directory, (success_key, safety_key) in POLICY_KEYS.items():
        summary = load_json(result_directory / directory / "summary.json")
        if int(summary["episodes"]) != episode_count:
            failures.append(f"{directory}: episode count disagrees with comparison.json")
        if abs(float(summary["success_rate"]) - float(comparison[success_key])) > 1e-12:
            failures.append(f"{directory}: success rate disagrees with comparison.json")
        if int(summary["safety_violations"]) != int(comparison[safety_key]):
            failures.append(f"{directory}: safety violations disagree with comparison.json")

    learned_success = float(comparison["learned_success_rate"])
    learned_safety = int(comparison["learned_safety_violations"])
    actual_non_regression = learned_success >= float(
        comparison["baseline_success_rate"]
    ) and learned_safety <= int(comparison["baseline_safety_violations"])
    if bool(comparison["non_regression"]) != actual_non_regression:
        failures.append("non_regression flag disagrees with recorded metrics")
    if require_non_regression and not actual_non_regression:
        failures.append("learned policy regressed against the zero-residual baseline")
    if learned_success < min_learned_success_rate:
        failures.append(
            f"learned success rate {learned_success:.3f} < required {min_learned_success_rate:.3f}"
        )
    if max_learned_safety_violations is not None and learned_safety > max_learned_safety_violations:
        failures.append(
            f"learned safety violations {learned_safety} > allowed {max_learned_safety_violations}"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate consistency and acceptance thresholds for residual-RL results."
    )
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--min-evaluation-episodes", type=int, default=1)
    parser.add_argument("--min-learned-success-rate", type=float, default=0.0)
    parser.add_argument("--max-learned-safety-violations", type=int)
    parser.add_argument("--require-non-regression", action="store_true")
    args = parser.parse_args()

    failures = validate_results(
        args.result_directory,
        min_evaluation_episodes=args.min_evaluation_episodes,
        min_learned_success_rate=args.min_learned_success_rate,
        max_learned_safety_violations=args.max_learned_safety_violations,
        require_non_regression=args.require_non_regression,
    )
    if failures:
        raise SystemExit("residual-RL acceptance failed: " + "; ".join(failures))
    print("residual-RL acceptance passed")


if __name__ == "__main__":
    main()
