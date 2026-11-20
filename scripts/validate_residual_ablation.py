from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_ablation(
    summary: dict,
    min_seeds: int,
    min_evaluation_episodes: int,
    min_teacher_success_rate: float,
    max_teacher_safety_violations: int | None,
    require_teacher_non_regression: bool,
) -> list[str]:
    failures: list[str] = []
    seeds = set(int(seed) for seed in summary["seeds"])
    if len(seeds) < min_seeds:
        failures.append(f"seed count {len(seeds)} < required {min_seeds}")

    runs = summary["runs"]
    identities = {(int(run["seed"]), run["rehearsal_mode"]) for run in runs}
    expected = {(seed, mode) for seed in seeds for mode in ("none", "teacher")}
    if identities != expected or len(runs) != len(expected):
        failures.append("every seed must have exactly one none and one teacher run")
    for run in runs:
        if int(run["evaluation_episodes"]) < min_evaluation_episodes:
            failures.append(
                f"seed {run['seed']} {run['rehearsal_mode']}: too few evaluation episodes"
            )
        if (
            require_teacher_non_regression
            and run["rehearsal_mode"] == "teacher"
            and not run["non_regression"]
        ):
            failures.append(f"seed {run['seed']}: teacher-rehearsal policy regressed")

    policies = {policy["policy"]: policy for policy in summary["policies"]}
    required_policies = {
        "baseline",
        "behavior_cloned",
        "learned_none",
        "learned_teacher",
        "pose_reference",
    }
    if set(policies) != required_policies:
        failures.append("aggregate policy set is incomplete")
        return failures

    teacher = policies["learned_teacher"]
    teacher_success = float(teacher["success_rate"])
    teacher_safety = int(teacher["safety_violations"])
    if teacher_success < min_teacher_success_rate:
        failures.append(
            f"teacher-rehearsal success {teacher_success:.3f} < required "
            f"{min_teacher_success_rate:.3f}"
        )
    if max_teacher_safety_violations is not None and teacher_safety > max_teacher_safety_violations:
        failures.append(
            f"teacher-rehearsal safety violations {teacher_safety} > allowed "
            f"{max_teacher_safety_violations}"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate structure and acceptance thresholds for residual ablation results."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--min-evaluation-episodes", type=int, default=1)
    parser.add_argument("--min-teacher-success-rate", type=float, default=0.0)
    parser.add_argument("--max-teacher-safety-violations", type=int)
    parser.add_argument("--require-teacher-non-regression", action="store_true")
    args = parser.parse_args()

    failures = validate_ablation(
        load_json(args.summary),
        min_seeds=args.min_seeds,
        min_evaluation_episodes=args.min_evaluation_episodes,
        min_teacher_success_rate=args.min_teacher_success_rate,
        max_teacher_safety_violations=args.max_teacher_safety_violations,
        require_teacher_non_regression=args.require_teacher_non_regression,
    )
    if failures:
        raise SystemExit("residual ablation acceptance failed: " + "; ".join(failures))
    print("residual ablation acceptance passed")


if __name__ == "__main__":
    main()
