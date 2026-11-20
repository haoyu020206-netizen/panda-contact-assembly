from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

POLICY_ORDER = (
    ("baseline", "Zero residual", "#7f8c8d"),
    ("behavior_cloned", "Behavior cloned", "#3498db"),
    ("learned_none", "SAC, no rehearsal", "#e67e22"),
    ("learned_teacher", "SAC + rehearsal", "#2ecc71"),
    ("pose_reference", "Pose reference", "#9b59b6"),
)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover_runs(root: Path) -> list[dict]:
    runs: list[dict] = []
    for comparison_path in sorted(root.rglob("comparison.json")):
        comparison = load_json(comparison_path)
        mode = comparison.get("rehearsal_mode")
        if mode not in {"none", "teacher"}:
            continue
        directory = comparison_path.parent
        summaries = {
            name: load_json(directory / name / "summary.json")
            for name in ("baseline", "behavior-cloned", "learned", "pose-reference")
        }
        runs.append(
            {
                "seed": int(comparison["seed"]),
                "rehearsal_mode": mode,
                "timesteps": int(comparison["timesteps"]),
                "evaluation_episodes": int(comparison["evaluation_episodes"]),
                "non_regression": bool(comparison["non_regression"]),
                "summaries": summaries,
                "source": str(comparison_path),
            }
        )
    if not runs:
        raise ValueError(f"no ablation comparison.json files found under {root}")
    identities = [(run["seed"], run["rehearsal_mode"]) for run in runs]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate seed/rehearsal-mode result sets")
    return runs


def select_unique_seed_runs(runs: list[dict]) -> list[dict]:
    selected: dict[int, dict] = {}
    for run in runs:
        seed = run["seed"]
        if seed not in selected or run["rehearsal_mode"] == "teacher":
            selected[seed] = run
    return [selected[seed] for seed in sorted(selected)]


def summarize_policy(name: str, run_summaries: Iterable[dict]) -> dict:
    summaries = list(run_summaries)
    episodes = sum(int(summary["episodes"]) for summary in summaries)
    successes = sum(int(summary["successes"]) for summary in summaries)
    weighted_peak_force = sum(
        float(summary["mean_peak_force_n"]) * int(summary["episodes"]) for summary in summaries
    )
    seed_rates = [float(summary["success_rate"]) for summary in summaries]
    return {
        "policy": name,
        "seeds": len(summaries),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "success_rate_seed_mean": float(np.mean(seed_rates)),
        "success_rate_seed_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
        "safety_violations": sum(int(summary["safety_violations"]) for summary in summaries),
        "mean_peak_force_n": weighted_peak_force / episodes,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    seed_runs = select_unique_seed_runs(runs)
    none_runs = sorted(
        (run for run in runs if run["rehearsal_mode"] == "none"),
        key=lambda run: run["seed"],
    )
    teacher_runs = sorted(
        (run for run in runs if run["rehearsal_mode"] == "teacher"),
        key=lambda run: run["seed"],
    )
    policies = [
        summarize_policy("baseline", (run["summaries"]["baseline"] for run in seed_runs)),
        summarize_policy(
            "behavior_cloned",
            (run["summaries"]["behavior-cloned"] for run in seed_runs),
        ),
    ]
    if none_runs:
        policies.append(
            summarize_policy(
                "learned_none",
                (run["summaries"]["learned"] for run in none_runs),
            )
        )
    if teacher_runs:
        policies.append(
            summarize_policy(
                "learned_teacher",
                (run["summaries"]["learned"] for run in teacher_runs),
            )
        )
    policies.append(
        summarize_policy(
            "pose_reference",
            (run["summaries"]["pose-reference"] for run in seed_runs),
        )
    )
    return {
        "seeds": sorted({run["seed"] for run in runs}),
        "run_count": len(runs),
        "timesteps": sorted({run["timesteps"] for run in runs}),
        "evaluation_episodes": sorted({run["evaluation_episodes"] for run in runs}),
        "policies": policies,
        "runs": runs,
    }


def write_run_csv(runs: list[dict], path: Path) -> None:
    fields = (
        "seed",
        "rehearsal_mode",
        "timesteps",
        "evaluation_episodes",
        "non_regression",
        "baseline_success_rate",
        "bc_success_rate",
        "learned_success_rate",
        "learned_safety_violations",
        "learned_mean_peak_force_n",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in sorted(runs, key=lambda item: (item["seed"], item["rehearsal_mode"])):
            summaries = run["summaries"]
            writer.writerow(
                {
                    "seed": run["seed"],
                    "rehearsal_mode": run["rehearsal_mode"],
                    "timesteps": run["timesteps"],
                    "evaluation_episodes": run["evaluation_episodes"],
                    "non_regression": run["non_regression"],
                    "baseline_success_rate": summaries["baseline"]["success_rate"],
                    "bc_success_rate": summaries["behavior-cloned"]["success_rate"],
                    "learned_success_rate": summaries["learned"]["success_rate"],
                    "learned_safety_violations": summaries["learned"]["safety_violations"],
                    "learned_mean_peak_force_n": summaries["learned"]["mean_peak_force_n"],
                }
            )


def plot_summary(aggregate: dict, path: Path) -> None:
    indexed = {policy["policy"]: policy for policy in aggregate["policies"]}
    available = [entry for entry in POLICY_ORDER if entry[0] in indexed]
    labels = [label for _, label, _ in available]
    colors = [color for _, _, color in available]
    policies = [indexed[key] for key, _, _ in available]
    panels = (
        ("success_rate", "Success rate", "%", 100.0),
        ("safety_violations", "Safety terminations", "episodes", 1.0),
        ("mean_peak_force_n", "Mean peak contact force", "N", 1.0),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    for axis, (field, title, unit, scale) in zip(axes, panels, strict=True):
        values = [float(policy[field]) * scale for policy in policies]
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=24)
        upper = max(values) * 1.2 + (2.0 if unit == "%" else 0.5)
        axis.set_ylim(0.0, max(upper, 1.0))
        for bar, value in zip(bars, values, strict=True):
            label = f"{value:.0f}" if unit in {"%", "episodes"} else f"{value:.1f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + axis.get_ylim()[1] * 0.018,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Residual-policy multi-seed rehearsal ablation", fontsize=14)
    figure.text(
        0.5,
        0.01,
        (
            f"seeds={aggregate['seeds']} | steps={aggregate['timesteps']} | "
            f"evaluation episodes/run={aggregate['evaluation_episodes']} | simulated forces"
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed residual-policy ablations.")
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.input_directory)
    aggregate = aggregate_runs(runs)
    with (args.output / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    write_run_csv(runs, args.output / "ablation_runs.csv")
    plot_summary(aggregate, args.output / "ablation_summary.png")
    print(json.dumps({key: aggregate[key] for key in ("seeds", "run_count")}, indent=2))


if __name__ == "__main__":
    main()
