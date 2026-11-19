from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from panda_assembly.config import load_config
from panda_assembly.residual_rl import (
    ResidualPegInsertionEnv,
    evaluate_residual_policy,
    save_episode_results,
    zero_residual_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a bounded SAC residual on impedance insertion."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--output", type=Path, default=Path("runs/residual-sac"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.evaluation_episodes <= 0:
        raise SystemExit("timesteps and evaluation episodes must be positive")

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
    except ImportError as error:
        raise SystemExit('Install training dependencies with: pip install -e ".[rl]"') from error

    config = load_config(args.config)
    seed = config.residual_rl.seed if args.seed is None else args.seed
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = evaluate_residual_policy(
        config,
        zero_residual_policy,
        args.evaluation_episodes,
        seed=seed,
    )
    save_episode_results(baseline, args.output / "baseline")

    env = ResidualPegInsertionEnv(config)
    check_env(env, warn=True, skip_render_check=True)
    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=max(10_000, args.timesteps),
        learning_starts=min(1_000, max(100, args.timesteps // 10)),
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.output / "residual_sac")
    env.close()

    def learned_policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    learned = evaluate_residual_policy(
        config,
        learned_policy,
        args.evaluation_episodes,
        seed=seed,
    )
    save_episode_results(learned, args.output / "learned")

    baseline_success = sum(result.success for result in baseline) / len(baseline)
    learned_success = sum(result.success for result in learned) / len(learned)
    baseline_safety = sum(result.safety_violation for result in baseline)
    learned_safety = sum(result.safety_violation for result in learned)
    comparison = {
        "algorithm": "SAC",
        "timesteps": args.timesteps,
        "evaluation_episodes": args.evaluation_episodes,
        "seed": seed,
        "baseline_success_rate": baseline_success,
        "learned_success_rate": learned_success,
        "success_rate_delta": learned_success - baseline_success,
        "baseline_safety_violations": baseline_safety,
        "learned_safety_violations": learned_safety,
        "non_regression": learned_success >= baseline_success and learned_safety <= baseline_safety,
    }
    with (args.output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
