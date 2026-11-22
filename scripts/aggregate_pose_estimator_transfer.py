from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CONDITION_ORDER = (
    "oracle",
    "camera-nominal",
    "camera-latency",
    "camera-dropout",
    "camera-correlated",
    "camera-combined",
)
POLICY_ORDER = ("learned", "learned-search")


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


def paired_outcomes(run_directory: Path, condition_key: str) -> dict[str, int]:
    policy_rows = {}
    for policy_key in POLICY_ORDER:
        path = run_directory / condition_key / policy_key / "episodes.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            policy_rows[policy_key] = list(csv.DictReader(handle))
    learned = policy_rows["learned"]
    searched = policy_rows["learned-search"]
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


def discover_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.rglob("pose_estimator_summary.json")):
        summary = load_json(summary_path)
        for condition_key in CONDITION_ORDER:
            summary["conditions"][condition_key]["paired_outcomes"] = paired_outcomes(
                summary_path.parent,
                condition_key,
            )
        summary["source"] = summary_path.relative_to(root).as_posix()
        runs.append(summary)
    if not runs:
        raise ValueError(f"no pose_estimator_summary.json files found under {root}")
    seeds = [int(run["seed"]) for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate pose-estimator seed results")
    return sorted(runs, key=lambda run: int(run["seed"]))


def aggregate_policy(summaries: list[dict]) -> dict:
    episodes = sum(int(summary["episodes"]) for summary in summaries)
    samples = sum(int(summary["pose_estimator_samples"]) for summary in summaries)
    dropouts = sum(int(summary["pose_estimator_dropouts"]) for summary in summaries)

    def weighted(field: str) -> float:
        return (
            sum(float(summary[field]) * int(summary["episodes"]) for summary in summaries)
            / episodes
        )

    successes = sum(int(summary["successes"]) for summary in summaries)
    contacts = sum(int(summary["contact_episodes"]) for summary in summaries)
    seed_success_rates = [float(summary["success_rate"]) for summary in summaries]
    return {
        "seeds": len(summaries),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "success_rate_seed_mean": float(np.mean(seed_success_rates)),
        "success_rate_seed_std": (
            float(np.std(seed_success_rates, ddof=1)) if len(seed_success_rates) > 1 else 0.0
        ),
        "safety_violations": sum(int(summary["safety_violations"]) for summary in summaries),
        "mean_peak_force_n": weighted("mean_peak_force_n"),
        "mean_reward": weighted("mean_reward"),
        "contact_episodes": contacts,
        "contact_rate": contacts / episodes,
        "mean_contact_steps": weighted("mean_contact_steps"),
        "search_triggers": sum(int(summary["search_triggers"]) for summary in summaries),
        "recovery_successes": sum(int(summary["recovery_successes"]) for summary in summaries),
        "mean_pose_estimate_age_s": weighted("mean_pose_estimate_age_s"),
        "max_pose_estimate_age_s": max(
            float(summary["max_pose_estimate_age_s"]) for summary in summaries
        ),
        "pose_estimator_samples": samples,
        "pose_estimator_dropouts": dropouts,
        "pose_estimator_delivered_updates": sum(
            int(summary["pose_estimator_delivered_updates"]) for summary in summaries
        ),
        "pose_estimator_dropout_rate": dropouts / samples if samples else 0.0,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    expected_conditions = set(CONDITION_ORDER)
    for run in runs:
        if set(run["conditions"]) != expected_conditions:
            raise ValueError(f"seed {run['seed']} has unexpected estimator conditions")
    conditions: dict[str, dict] = {}
    for condition_key in CONDITION_ORDER:
        seed_conditions = [run["conditions"][condition_key] for run in runs]
        estimator_config = seed_conditions[0]["pose_estimator"]
        if any(
            condition["pose_estimator"] != estimator_config for condition in seed_conditions[1:]
        ):
            raise ValueError(f"{condition_key} estimator configuration differs across seeds")
        policies = {
            policy_key: aggregate_policy(
                [condition["policies"][policy_key] for condition in seed_conditions]
            )
            for policy_key in POLICY_ORDER
        }
        paired = {
            field: sum(int(condition["paired_outcomes"][field]) for condition in seed_conditions)
            for field in ("rescued", "harmed", "both_success", "neither")
        }
        conditions[condition_key] = {
            "label": seed_conditions[0]["label"],
            "pose_estimator": estimator_config,
            "policies": policies,
            "paired_outcomes": paired,
        }
    oracle_success = conditions["oracle"]["policies"]["learned"]["success_rate"]
    for condition in conditions.values():
        condition["learned_success_delta_vs_oracle"] = (
            condition["policies"]["learned"]["success_rate"] - oracle_success
        )
    return {
        "seeds": [int(run["seed"]) for run in runs],
        "run_count": len(runs),
        "episodes_per_seed": sorted({int(run["episodes"]) for run in runs}),
        "socket_clearance_m": sorted({float(run["socket_clearance_m"]) for run in runs}),
        "conditions": conditions,
        "runs": runs,
    }


def write_runs_csv(runs: list[dict], path: Path) -> None:
    fields = (
        "seed",
        "condition",
        "policy",
        "episodes",
        "successes",
        "success_rate",
        "safety_violations",
        "mean_peak_force_n",
        "contact_rate",
        "mean_pose_estimate_age_s",
        "max_pose_estimate_age_s",
        "pose_estimator_samples",
        "pose_estimator_dropouts",
        "pose_estimator_dropout_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for condition_key in CONDITION_ORDER:
                for policy_key in POLICY_ORDER:
                    summary = run["conditions"][condition_key]["policies"][policy_key]
                    writer.writerow(
                        {
                            "seed": run["seed"],
                            "condition": condition_key,
                            "policy": policy_key,
                            **{field: summary[field] for field in fields[3:]},
                        }
                    )


def plot_summary(aggregate: dict, path: Path) -> None:
    conditions = [aggregate["conditions"][key] for key in CONDITION_ORDER]
    labels = [condition["label"] for condition in conditions]
    x_positions = np.arange(len(conditions), dtype=float)
    figure, axes = plt.subplots(1, 4, figsize=(19.0, 4.8))
    policy_styles = (
        ("learned", "MVP-trained", "#2ecc71", -0.18),
        ("learned-search", "MVP + search", "#e67e22", 0.18),
    )
    for policy_key, label, color, offset in policy_styles:
        success = [
            condition["policies"][policy_key]["success_rate"] * 100 for condition in conditions
        ]
        safety = [
            condition["policies"][policy_key]["safety_violations"] for condition in conditions
        ]
        axes[0].bar(x_positions + offset, success, width=0.34, label=label, color=color)
        axes[1].bar(x_positions + offset, safety, width=0.34, label=label, color=color)
    learned = [condition["policies"]["learned"] for condition in conditions]
    axes[2].bar(
        x_positions,
        [policy["mean_pose_estimate_age_s"] * 1000 for policy in learned],
        color="#3498db",
    )
    axes[3].bar(
        x_positions,
        [policy["pose_estimator_dropout_rate"] * 100 for policy in learned],
        color="#9b59b6",
    )
    panels = (
        ("Success rate", "%", 105.0),
        ("Safety terminations", "episodes", None),
        ("Mean estimate age", "ms", None),
        ("Observed frame dropout", "%", 40.0),
    )
    for axis, (title, unit, fixed_upper) in zip(axes, panels, strict=True):
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(x_positions, labels, rotation=22, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if fixed_upper is not None:
            axis.set_ylim(0.0, fixed_upper)
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("Explicit pose-estimator transfer on 3 mm Menagerie insertion", fontsize=14)
    figure.text(
        0.5,
        0.01,
        (
            f"seeds={aggregate['seeds']} | episodes/seed={aggregate['episodes_per_seed']} | "
            "sampled, delayed, held estimates | simulated forces"
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed pose-estimator transfer.")
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.input_directory)
    aggregate = aggregate_runs(runs)
    with (args.output / "pose_estimator_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    write_runs_csv(runs, args.output / "pose_estimator_runs.csv")
    plot_summary(aggregate, args.output / "pose_estimator_aggregate.png")
    print(json.dumps({"seeds": aggregate["seeds"], "run_count": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
