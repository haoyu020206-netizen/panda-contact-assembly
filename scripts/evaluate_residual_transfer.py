from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from panda_assembly.config import load_config
from panda_assembly.residual_rl import (
    evaluate_residual_policy,
    pose_servo_reference_policy,
    save_episode_results,
    zero_residual_policy,
)

POLICIES = (
    ("baseline", "Zero residual", "#7f8c8d"),
    ("learned", "MVP-trained policy", "#2ecc71"),
    ("pose-reference", "Pose reference", "#9b59b6"),
)


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_transfer(summary: dict, output: Path) -> None:
    condition_items = list(summary["conditions"].items())
    condition_labels = [condition["label"] for _, condition in condition_items]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    panels = (
        ("success_rate", "Success rate", "%", 100.0),
        ("safety_violations", "Safety terminations", "episodes", 1.0),
        ("mean_peak_force_n", "Mean peak contact force", "N", 1.0),
    )
    x_positions = np.arange(len(condition_items), dtype=float)
    width = 0.24
    for axis, (field, title, unit, scale) in zip(axes, panels, strict=True):
        all_values: list[float] = []
        for policy_index, (policy_key, policy_label, color) in enumerate(POLICIES):
            values = [
                float(condition["policies"][policy_key][field]) * scale
                for _, condition in condition_items
            ]
            all_values.extend(values)
            offsets = x_positions + (policy_index - 1) * width
            bars = axis.bar(offsets, values, width=width, label=policy_label, color=color)
            for bar, value in zip(bars, values, strict=True):
                value_label = f"{value:.0f}" if unit in {"%", "episodes"} else f"{value:.1f}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    value_label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(x_positions, condition_labels, rotation=18)
        axis.grid(axis="y", alpha=0.25)
        upper = 105.0 if unit == "%" else max(all_values) * 1.2 + 0.5
        axis.set_ylim(0.0, max(upper, 1.0))
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("Zero-shot residual-policy transfer to official Menagerie Panda", fontsize=14)
    figure.text(
        0.5,
        0.01,
        (
            f"seed={summary['seed']} | {summary['episodes']} paired episodes/condition | "
            "same MVP-trained policy | simulated forces"
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.94))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an MVP-trained residual policy on Menagerie and pose perturbations."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--target-config", type=Path, default=Path("configs/menagerie.toml"))
    parser.add_argument("--output", type=Path, default=Path("runs/residual-transfer"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--noise-std-m", type=float, default=0.001)
    parser.add_argument("--bias-x-m", type=float, default=0.002)
    parser.add_argument("--bias-y-m", type=float, default=-0.002)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")
    if args.noise_std_m < 0.0:
        raise SystemExit("noise standard deviation must be nonnegative")

    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit('Install transfer dependencies with: pip install -e ".[rl]"') from error

    source_config = load_config(args.source_config)
    target_config = load_config(args.target_config)
    seed = source_config.residual_rl.seed if args.seed is None else args.seed
    model = SAC.load(args.model)

    def learned_policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    conditions = (
        (
            "mvp-clean",
            "MVP clean",
            source_config,
            np.zeros(3),
            np.zeros(3),
        ),
        (
            "menagerie-clean",
            "Menagerie clean",
            target_config,
            np.zeros(3),
            np.zeros(3),
        ),
        (
            "menagerie-noisy",
            "Menagerie + 1 mm noise",
            target_config,
            np.full(3, args.noise_std_m),
            np.zeros(3),
        ),
        (
            "menagerie-biased",
            "Menagerie + pose bias",
            target_config,
            np.zeros(3),
            np.array([args.bias_x_m, args.bias_y_m, 0.0]),
        ),
    )
    policy_functions = (
        ("baseline", zero_residual_policy),
        ("learned", learned_policy),
        ("pose-reference", pose_servo_reference_policy),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    condition_summaries: dict[str, dict] = {}
    for key, label, config, noise_std, bias in conditions:
        policy_summaries: dict[str, dict] = {}
        for policy_key, policy in policy_functions:
            policy_output = args.output / key / policy_key
            results = evaluate_residual_policy(
                config,
                policy,
                args.episodes,
                seed=seed,
                observation_position_noise_std_m=noise_std,
                observation_position_bias_m=bias,
            )
            save_episode_results(results, policy_output)
            policy_summaries[policy_key] = load_summary(policy_output / "summary.json")
        condition_summaries[key] = {
            "label": label,
            "model_variant": config.model.variant,
            "observation_position_noise_std_m": noise_std.tolist(),
            "observation_position_bias_m": bias.tolist(),
            "policies": policy_summaries,
        }

    source_learned = condition_summaries["mvp-clean"]["policies"]["learned"]
    target_learned = condition_summaries["menagerie-clean"]["policies"]["learned"]
    summary = {
        "seed": seed,
        "episodes": args.episodes,
        "source_config": str(args.source_config),
        "target_config": str(args.target_config),
        "model": str(args.model),
        "conditions": condition_summaries,
        "zero_shot_success_rate_delta": (
            float(target_learned["success_rate"]) - float(source_learned["success_rate"])
        ),
        "zero_shot_peak_force_delta_n": (
            float(target_learned["mean_peak_force_n"]) - float(source_learned["mean_peak_force_n"])
        ),
    }
    with (args.output / "transfer_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_transfer(summary, args.output / "transfer_summary.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
