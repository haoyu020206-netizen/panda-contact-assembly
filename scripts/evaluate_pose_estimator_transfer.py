from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from panda_assembly.config import load_config
from panda_assembly.contact_search import SpiralResidualPolicy
from panda_assembly.pose_estimation import PoseEstimatorConfig
from panda_assembly.residual_rl import evaluate_residual_policy, save_episode_results

CONDITIONS = (
    (
        "oracle",
        "Ideal pose",
        PoseEstimatorConfig(update_period_s=0.002),
    ),
    (
        "camera-nominal",
        "25 Hz camera",
        PoseEstimatorConfig(
            update_period_s=0.04,
            latency_s=0.04,
            noise_std_m=(0.0005, 0.0005, 0.00025),
            correlation_time_s=0.2,
        ),
    ),
    (
        "camera-latency",
        "120 ms latency",
        PoseEstimatorConfig(
            update_period_s=0.04,
            latency_s=0.12,
            noise_std_m=(0.0005, 0.0005, 0.00025),
            correlation_time_s=0.2,
        ),
    ),
    (
        "camera-dropout",
        "30% dropout",
        PoseEstimatorConfig(
            update_period_s=0.04,
            latency_s=0.04,
            dropout_probability=0.30,
            noise_std_m=(0.0005, 0.0005, 0.00025),
            correlation_time_s=0.2,
        ),
    ),
    (
        "camera-correlated",
        "Correlated 1.5 mm error",
        PoseEstimatorConfig(
            update_period_s=0.04,
            latency_s=0.04,
            noise_std_m=(0.0015, 0.0015, 0.0005),
            correlation_time_s=0.8,
        ),
    ),
    (
        "camera-combined",
        "Latency + dropout + error",
        PoseEstimatorConfig(
            update_period_s=0.04,
            latency_s=0.12,
            dropout_probability=0.30,
            noise_std_m=(0.0015, 0.0015, 0.0005),
            correlation_time_s=0.8,
        ),
    ),
)


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate residual insertion with explicit delayed pose estimates."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/menagerie.toml"))
    parser.add_argument("--output", type=Path, default=Path("runs/pose-estimator-transfer"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--socket-clearance-m", type=float, default=0.003)
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

    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise SystemExit(
            'Install pose-estimator benchmark dependencies with: pip install -e ".[rl]"'
        ) from error

    config = load_config(args.config)
    if args.insert_time_s <= config.peg_task.align_time:
        raise SystemExit("insert time must be later than align time")
    if args.episode_duration_s <= args.insert_time_s:
        raise SystemExit("episode duration must be later than insert time")
    config = replace(
        config,
        peg_task=replace(config.peg_task, insert_time=args.insert_time_s),
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

    args.output.mkdir(parents=True, exist_ok=True)
    condition_summaries: dict[str, dict] = {}
    for condition_key, label, estimator_config in CONDITIONS:
        policy_summaries: dict[str, dict] = {}
        for policy_key, policy in (
            ("learned", learned_policy),
            ("learned-search", search_policy()),
        ):
            policy_output = args.output / condition_key / policy_key
            results = evaluate_residual_policy(
                config,
                policy,
                args.episodes,
                seed=seed,
                pose_estimator_config=estimator_config,
                socket_clearance_m=args.socket_clearance_m,
                contact_detection_force_n=args.contact_force_n,
            )
            save_episode_results(results, policy_output)
            policy_summaries[policy_key] = load_summary(policy_output / "summary.json")
        condition_summaries[condition_key] = {
            "label": label,
            "pose_estimator": asdict(estimator_config),
            "policies": policy_summaries,
        }

    oracle_success = float(condition_summaries["oracle"]["policies"]["learned"]["success_rate"])
    for condition in condition_summaries.values():
        condition["learned_success_delta_vs_oracle"] = (
            float(condition["policies"]["learned"]["success_rate"]) - oracle_success
        )
    summary = {
        "seed": seed,
        "episodes": args.episodes,
        "config": str(args.config),
        "model": str(args.model),
        "socket_clearance_m": args.socket_clearance_m,
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
    with (args.output / "pose_estimator_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"seed": seed, "conditions": list(condition_summaries)}, indent=2))


if __name__ == "__main__":
    main()
