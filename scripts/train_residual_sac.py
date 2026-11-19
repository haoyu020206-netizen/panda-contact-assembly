from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plot_residual_results import plot_residual_summary

from panda_assembly.config import load_config
from panda_assembly.residual_rl import (
    ResidualPegInsertionEnv,
    collect_reference_dataset,
    evaluate_residual_policy,
    pose_servo_reference_policy,
    save_episode_results,
    zero_residual_policy,
)


def behavior_clone_actor(model, observations, actions, epochs: int, seed: int) -> float:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    observation_tensor = torch.as_tensor(observations, device=model.device)
    action_tensor = torch.as_tensor(actions, device=model.device)
    optimizer = torch.optim.Adam(model.policy.actor.parameters(), lr=1e-3)
    batch_size = min(256, len(observations))
    final_loss = 0.0
    model.policy.actor.set_training_mode(True)
    for _ in range(epochs):
        order = torch.randperm(len(observations), generator=generator)
        for start in range(0, len(observations), batch_size):
            indices = order[start : start + batch_size].to(model.device)
            predicted = model.policy.actor(observation_tensor[indices], deterministic=True)
            loss = torch.nn.functional.mse_loss(predicted, action_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    model.policy.actor.set_training_mode(False)
    return final_loss


def make_bc_rehearsal_callback(observations, actions, seed: int):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    class BehaviorCloningRehearsalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.generator = torch.Generator(device="cpu").manual_seed(seed)
            self.observation_tensor = None
            self.action_tensor = None
            self.optimizer = None

        def _on_training_start(self) -> None:
            self.observation_tensor = torch.as_tensor(observations, device=self.model.device)
            self.action_tensor = torch.as_tensor(actions, device=self.model.device)
            self.optimizer = torch.optim.Adam(self.model.policy.actor.parameters(), lr=3e-4)

        def _on_step(self) -> bool:
            assert self.observation_tensor is not None
            assert self.action_tensor is not None
            assert self.optimizer is not None
            sample_count = min(256, len(observations))
            indices = torch.randint(
                len(observations),
                (sample_count,),
                generator=self.generator,
            ).to(self.model.device)
            actor = self.model.policy.actor
            predicted = actor(self.observation_tensor[indices], deterministic=True)
            loss = torch.nn.functional.mse_loss(predicted, self.action_tensor[indices])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.logger.record("train/bc_rehearsal_loss", float(loss.detach().cpu()))
            return True

    return BehaviorCloningRehearsalCallback()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a bounded SAC residual on impedance insertion."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--output", type=Path, default=Path("runs/residual-sac"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--bc-episodes", type=int, default=20)
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if min(args.timesteps, args.evaluation_episodes, args.bc_episodes, args.bc_epochs) <= 0:
        raise SystemExit("timesteps, evaluation episodes, and BC settings must be positive")

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
    pose_reference = evaluate_residual_policy(
        config,
        pose_servo_reference_policy,
        args.evaluation_episodes,
        seed=seed,
    )
    save_episode_results(pose_reference, args.output / "pose-reference")

    env = ResidualPegInsertionEnv(config)
    check_env(env, warn=True, skip_render_check=True)
    model = SAC(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=1e-5,
        buffer_size=max(10_000, args.timesteps),
        learning_starts=0,
        batch_size=256,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.0001",
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
    )
    reference_observations, reference_actions = collect_reference_dataset(
        config,
        args.bc_episodes,
        seed=seed + 10_000,
    )
    bc_final_loss = behavior_clone_actor(
        model,
        reference_observations,
        reference_actions,
        args.bc_epochs,
        seed,
    )
    model.save(args.output / "residual_bc")

    def current_policy(observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    behavior_cloned = evaluate_residual_policy(
        config,
        current_policy,
        args.evaluation_episodes,
        seed=seed,
    )
    save_episode_results(behavior_cloned, args.output / "behavior-cloned")

    rehearsal_callback = make_bc_rehearsal_callback(
        reference_observations,
        reference_actions,
        seed,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=rehearsal_callback,
        progress_bar=False,
    )
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
    bc_success = sum(result.success for result in behavior_cloned) / len(behavior_cloned)
    learned_success = sum(result.success for result in learned) / len(learned)
    baseline_safety = sum(result.safety_violation for result in baseline)
    bc_safety = sum(result.safety_violation for result in behavior_cloned)
    reference_success = sum(result.success for result in pose_reference) / len(pose_reference)
    reference_safety = sum(result.safety_violation for result in pose_reference)
    learned_safety = sum(result.safety_violation for result in learned)
    comparison = {
        "algorithm": "BC+SAC",
        "timesteps": args.timesteps,
        "evaluation_episodes": args.evaluation_episodes,
        "seed": seed,
        "bc_episodes": args.bc_episodes,
        "bc_epochs": args.bc_epochs,
        "bc_samples": len(reference_observations),
        "bc_final_loss": bc_final_loss,
        "online_learning_rate": 1e-5,
        "initial_entropy_coefficient": 0.0001,
        "bc_rehearsal": "one 256-sample teacher batch per step at 3e-4 learning rate",
        "baseline_success_rate": baseline_success,
        "behavior_cloned_success_rate": bc_success,
        "learned_success_rate": learned_success,
        "success_rate_delta": learned_success - baseline_success,
        "baseline_safety_violations": baseline_safety,
        "behavior_cloned_safety_violations": bc_safety,
        "pose_reference_success_rate": reference_success,
        "pose_reference_safety_violations": reference_safety,
        "learned_safety_violations": learned_safety,
        "non_regression": learned_success >= baseline_success and learned_safety <= baseline_safety,
    }
    with (args.output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    plot_residual_summary(args.output)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
