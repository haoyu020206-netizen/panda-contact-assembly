from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

POLICIES = (
    ("baseline", "Zero residual", "#7f8c8d"),
    ("behavior-cloned", "Behavior cloned", "#3498db"),
    ("learned", "BC + SAC", "#2ecc71"),
    ("pose-reference", "Pose reference", "#9b59b6"),
)


def load_summary(result_directory: Path, policy: str) -> dict[str, float | int]:
    with (result_directory / policy / "summary.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_residual_summary(result_directory: Path) -> Path:
    summaries = [load_summary(result_directory, policy) for policy, _, _ in POLICIES]
    with (result_directory / "comparison.json").open(encoding="utf-8") as handle:
        comparison = json.load(handle)

    labels = [label for _, label, _ in POLICIES]
    colors = [color for _, _, color in POLICIES]
    success_rates = [float(summary["success_rate"]) * 100.0 for summary in summaries]
    safety_violations = [int(summary["safety_violations"]) for summary in summaries]
    peak_forces = [float(summary["mean_peak_force_n"]) for summary in summaries]

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    panels = (
        (axes[0], success_rates, "Success rate", "%", (0.0, 105.0)),
        (
            axes[1],
            safety_violations,
            "Safety terminations",
            "episodes",
            (0.0, max(safety_violations) * 1.18 + 0.5),
        ),
        (
            axes[2],
            peak_forces,
            "Mean peak contact force",
            "N",
            (0.0, max(peak_forces) * 1.18 + 1.0),
        ),
    )
    for axis, values, title, unit, limits in panels:
        bars = axis.bar(labels, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=22)
        for bar, value in zip(bars, values, strict=True):
            label = f"{value:.0f}" if unit in {"%", "episodes"} else f"{value:.1f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (limits[1] * 0.018),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.suptitle("Bounded residual policy on paired randomized insertions", fontsize=14)
    figure.text(
        0.5,
        0.01,
        (
            f"seed={comparison['seed']} | {comparison['evaluation_episodes']} paired episodes | "
            f"{comparison['timesteps']:,} SAC steps | simulated forces"
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))
    output = result_directory / "residual_rl_summary.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a residual-RL policy comparison artifact.")
    parser.add_argument("result_directory", type=Path)
    args = parser.parse_args()
    output = plot_residual_summary(args.result_directory)
    print(output)


if __name__ == "__main__":
    main()
