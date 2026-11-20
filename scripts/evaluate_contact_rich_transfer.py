from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from panda_assembly.config import load_config
from panda_assembly.contact_search import SpiralResidualPolicy
from panda_assembly.residual_rl import (
    evaluate_residual_policy,
    pose_servo_reference_policy,
    save_episode_results,
    zero_residual_policy,
)

POLICIES = (
    ("baseline", "Zero residual", "#7f8c8d"),
    ("learned", "MVP-trained", "#2ecc71"),
    ("learned-search", "MVP + spiral recovery", "#e67e22"),
    ("pose-reference", "Pose reference", "#9b59b6"),
)


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_contact_rich(summary: dict, output: Path) -> None:
    condition_items = list(summary["conditions"].items())
    condition_labels = [condition["label"] for _, condition in condition_items]
    panels = (
        ("success_rate", "Success rate", "%", 100.0),
        ("contact_rate", "Episodes with contact", "%", 100.0),
        ("safety_violations", "Safety terminations", "episodes", 1.0),
        ("recovery_successes", "Recovery successes", "episodes", 1.0),
        ("mean_peak_force_n", "Mean peak force", "N", 1.0),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(22.0, 4.8))
    x_positions = np.arange(len(condition_items), dtype=float)
    width = 0.19
    for axis, (field, title, unit, scale) in zip(axes, panels, strict=True):
        all_values: list[float] = []
        for policy_index, (policy_key, policy_label, color) in enumerate(POLICIES):
            values = [
                float(condition["policies"][policy_key][field]) * scale
                for _, condition in condition_items
            ]
            all_values.extend(values)
            offsets = x_positions + (policy_index - 1.5) * width
            bars = axis.bar(offsets, values, width=width, label=policy_label, color=color)
            for bar, value in zip(bars, values, strict=True):
                value_label = f"{value:.0f}" if unit in {"%", "episodes"} else f"{value:.1f}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    value_label,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(x_positions, condition_labels, rotation=18)
        axis.grid(axis="y", alpha=0.25)
        upper = 105.0 if unit == "%" else max(all_values) * 1.2 + 0.5
        axis.set_ylim(0.0, max(upper, 1.0))
    axes[0].legend(loc="lower left", fontsize=7)
    figure.suptitle("Contact-rich residual-policy transfer on Menagerie Panda", fontsize=14)
    figure.text(
        0.5,
        0.01,
        (
            f"seed={summary['seed']} | {summary['episodes']} paired episodes/condition | "
            "tight-clearance transfer | simulated forces"
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
        description="Evaluate narrow-clearance contact and spiral recovery on Menagerie."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/menagerie.toml"))
    parser.add_argument("--output", type=Path, default=Path("runs/contact-rich-transfer"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--socket-clearance-m", type=float, default=0.003)
    parser.add_argument("--bias-x-m", type=float, default=0.004)
    parser.add_argument("--bias-y-m", type=float, default=-0.004)
    parser.add_argument("--contact-force-n", type=float, default=1.0)
    parser.add_argument("--search-trigger-force-n", type=float, default=5.0)
    parser.add_argument("--search-radius-m", type=float, default=0.007)
    parser.add_argument("--episode-duration-s", type=float, default=18.0)
    parser.add_argument("--insert-time-s", type=float, default=12.0)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise SystemExit("episodes must be positive")
    if args.socket_clearance_m <= 0.0:
        raise SystemExit("socket clearance must be positive")
    if args.episode_duration_s <= 0.0:
        raise SystemExit("episode duration must be positive")
    if args.insert_time_s <= 0.0:
        raise SystemExit("insert time must be positive")

    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit(
            'Install contact-rich dependencies with: pip install -e ".[rl]"'
        ) from error

    config = load_config(args.config)
    if args.insert_time_s <= config.peg_task.align_time:
        raise SystemExit("insert time must be later than align time")
    if args.episode_duration_s <= args.insert_time_s:
        raise SystemExit("episode duration must be later than insert time")
    config = replace(
        config,
        peg_task=replace(
            config.peg_task,
            insert_time=args.insert_time_s,
        ),
        residual_rl=replace(
            config.residual_rl,
            episode_duration_s=args.episode_duration_s,
        ),
    )
    seed = config.residual_rl.seed if args.seed is None else args.seed
    model = SAC.load(args.model)

    def learned_policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def search_policy() -> SpiralResidualPolicy:
        return SpiralResidualPolicy(
            learned_policy,
            max_position_offset_m=np.asarray(config.residual_rl.max_position_offset_m),
            episode_duration_s=config.residual_rl.episode_duration_s,
            force_scale_n=config.force_safety.force_limit_n,
            trigger_force_n=args.search_trigger_force_n,
            search_radius_m=args.search_radius_m,
            entry_depth_m=config.contact_recovery.entry_depth_m,
        )

    conditions = (
        ("wide-clean", "5 mm clean", None, np.zeros(3)),
        (
            "narrow-clean",
            f"{args.socket_clearance_m * 1000:.0f} mm clean",
            args.socket_clearance_m,
            np.zeros(3),
        ),
        (
            "narrow-biased",
            f"{args.socket_clearance_m * 1000:.0f} mm + pose bias",
            args.socket_clearance_m,
            np.array([args.bias_x_m, args.bias_y_m, 0.0]),
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    condition_summaries: dict[str, dict] = {}
    for key, label, socket_clearance_m, bias in conditions:
        policy_functions = (
            ("baseline", zero_residual_policy),
            ("learned", learned_policy),
            ("learned-search", search_policy()),
            ("pose-reference", pose_servo_reference_policy),
        )
        policy_summaries: dict[str, dict] = {}
        for policy_key, policy in policy_functions:
            policy_output = args.output / key / policy_key
            results = evaluate_residual_policy(
                config,
                policy,
                args.episodes,
                seed=seed,
                observation_position_bias_m=bias,
                socket_clearance_m=socket_clearance_m,
                contact_detection_force_n=args.contact_force_n,
            )
            save_episode_results(results, policy_output)
            policy_summaries[policy_key] = load_summary(policy_output / "summary.json")
        condition_summaries[key] = {
            "label": label,
            "model_variant": config.model.variant,
            "socket_clearance_m": (
                config.peg_task.clearance if socket_clearance_m is None else socket_clearance_m
            ),
            "observation_position_bias_m": bias.tolist(),
            "policies": policy_summaries,
        }

    summary = {
        "seed": seed,
        "episodes": args.episodes,
        "config": str(args.config),
        "model": str(args.model),
        "contact_detection_force_n": args.contact_force_n,
        "search": {
            "trigger_force_n": args.search_trigger_force_n,
            "radius_m": args.search_radius_m,
            "entry_depth_m": config.contact_recovery.entry_depth_m,
        },
        "insert_time_s": args.insert_time_s,
        "episode_duration_s": args.episode_duration_s,
        "conditions": condition_summaries,
    }
    with (args.output / "contact_rich_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_contact_rich(summary, args.output / "contact_rich_summary.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
