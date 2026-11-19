from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from panda_assembly.config import ProjectConfig
from panda_assembly.controllers import apply_torque, site_jacobian, site_pose
from panda_assembly.experiments import (
    Variation,
    _cartesian_controller,
    _configure_variation,
    _site_offset_in_frame,
)
from panda_assembly.math_utils import interpolate_rotation, smoothstep, vertical_tool_frame
from panda_assembly.metrics import minimum_geom_proximity, total_contact_force
from panda_assembly.model import load_model, reset_to_home

FloatArray = NDArray[np.float64]
Policy = Callable[[NDArray[np.float32]], NDArray[np.float32]]


@dataclass
class ResidualActionFilter:
    """Rate-limit bounded normalized target-offset actions."""

    max_offset_m: FloatArray
    max_rate_m_s: FloatArray
    offset_m: FloatArray | None = None

    def reset(self) -> FloatArray:
        self.offset_m = np.zeros(3, dtype=float)
        return self.offset_m.copy()

    def command(self, action: FloatArray, interval_s: float, enabled: bool) -> FloatArray:
        if self.offset_m is None:
            self.reset()
        assert self.offset_m is not None
        if not enabled:
            return self.offset_m.copy()
        normalized = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if normalized.shape != (3,):
            raise ValueError(f"residual action must have shape (3,), got {normalized.shape}")
        desired_offset = normalized * self.max_offset_m
        maximum_step = self.max_rate_m_s * interval_s
        self.offset_m += np.clip(desired_offset - self.offset_m, -maximum_step, maximum_step)
        self.offset_m[:] = np.clip(self.offset_m, -self.max_offset_m, self.max_offset_m)
        return self.offset_m.copy()


@dataclass(frozen=True)
class ResidualEpisodeResult:
    episode: int
    success: bool
    safety_violation: bool
    total_reward: float
    steps: int
    completion_time_s: float | None
    max_contact_force_n: float
    final_xy_error_m: float
    final_insertion_depth_m: float
    lateral_offset_x: float
    lateral_offset_y: float
    friction_scale: float
    tool_mass_scale: float
    final_residual_x_m: float
    final_residual_y_m: float
    final_residual_z_m: float


class ResidualPegInsertionEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Residual Cartesian target learning on top of the impedance controller.

    The policy commands a bounded residual velocity. It cannot change impedance gains,
    orientation control, actuator limits, the force termination, or the nominal trajectory.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config
        self.rl = config.residual_rl
        self.model = load_model(
            timestep=config.simulation.timestep,
            variant=config.model.variant,
        )
        self.data = mujoco.MjData(self.model)
        reset_to_home(self.model, self.data)
        self.tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self.base_tool_mass = float(self.model.body_mass[self.tool_body_id])
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
        self.socket_wall_geom_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("socket_x_pos", "socket_x_neg", "socket_y_pos", "socket_y_neg")
        )
        self.action_repeat = round(self.rl.control_interval_s / self.model.opt.timestep)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(12,), dtype=np.float32)
        self.action_filter = ResidualActionFilter(
            max_offset_m=np.asarray(self.rl.max_position_offset_m, dtype=float),
            max_rate_m_s=np.asarray(self.rl.max_position_rate_m_s, dtype=float),
        )
        self.controller = _cartesian_controller(config, self.data.qpos[:7])
        self.variation = Variation()
        self.ee_id = -1
        self.peg_id = -1
        self.hole_id = -1
        self.start_position = np.zeros(3)
        self.start_rotation = np.eye(3)
        self.vertical_rotation = np.eye(3)
        self.hole_position = np.zeros(3)
        self.aligned_position = np.zeros(3)
        self.final_target = np.zeros(3)
        self.previous_xy_error_m = 0.0
        self.previous_depth_m = 0.0
        self.maximum_force_n = 0.0
        self.completion_time_s: float | None = None
        self.episode_done = False

    def _sample_variation(self, options: dict[str, Any] | None) -> Variation:
        if options and "variation" in options:
            supplied = options["variation"]
            if isinstance(supplied, Variation):
                return supplied
            if isinstance(supplied, dict):
                return Variation(**supplied)
            raise TypeError("options['variation'] must be a Variation or mapping")
        limit = self.rl.lateral_offset_limit_m
        return Variation(
            lateral_offset_x=float(self.np_random.uniform(-limit, limit)),
            lateral_offset_y=float(self.np_random.uniform(-limit, limit)),
            friction_scale=float(
                self.np_random.uniform(
                    self.rl.friction_scale_min,
                    self.rl.friction_scale_max,
                )
            ),
            tool_mass_scale=float(
                self.np_random.uniform(
                    self.rl.tool_mass_scale_min,
                    self.rl.tool_mass_scale_max,
                )
            ),
        )

    def _configure_episode(self) -> None:
        self.model.body_mass[self.tool_body_id] = self.base_tool_mass
        self.ee_id, self.peg_id, self.hole_id = _configure_variation(
            self.model,
            self.data,
            self.variation,
            self.config.peg_task.nominal_friction,
        )
        self.start_position, self.start_rotation = site_pose(self.model, self.data, self.ee_id)
        self.hole_position, _ = site_pose(self.model, self.data, self.hole_id)
        commanded_hole = self.hole_position.copy()
        commanded_hole[0] -= self.variation.lateral_offset_x
        commanded_hole[1] -= self.variation.lateral_offset_y
        self.vertical_rotation = vertical_tool_frame(self.start_rotation)
        tip_offset_local = _site_offset_in_frame(
            self.model,
            self.data,
            self.ee_id,
            self.tip_id,
        )
        vertical_tip_offset = self.vertical_rotation @ tip_offset_local
        self.aligned_position = np.array(
            [
                commanded_hole[0] - vertical_tip_offset[0],
                commanded_hole[1] - vertical_tip_offset[1],
                self.start_position[2],
            ]
        )
        self.final_target = self.aligned_position.copy()
        self.final_target[2] = (
            commanded_hole[2] - self.config.peg_task.target_depth - vertical_tip_offset[2]
        )
        self.controller = _cartesian_controller(self.config, self.data.qpos[:7])

    def _nominal_target(self, sim_time: float) -> tuple[FloatArray, FloatArray]:
        peg = self.config.peg_task
        if sim_time <= peg.approach_time:
            return self.start_position.copy(), self.start_rotation.copy()
        if sim_time <= peg.align_time:
            progress = (sim_time - peg.approach_time) / (peg.align_time - peg.approach_time)
            position = self.start_position + smoothstep(progress) * (
                self.aligned_position - self.start_position
            )
            rotation = interpolate_rotation(
                self.start_rotation,
                self.vertical_rotation,
                progress,
            )
            return position, rotation
        progress = (sim_time - peg.align_time) / (peg.insert_time - peg.align_time)
        position = self.aligned_position + smoothstep(progress) * (
            self.final_target - self.aligned_position
        )
        return position, self.vertical_rotation.copy()

    def _metrics(self) -> tuple[float, float, float, float]:
        tip_position, _ = site_pose(self.model, self.data, self.tip_id)
        xy_error = float(np.linalg.norm(tip_position[:2] - self.hole_position[:2]))
        depth = float(self.hole_position[2] - tip_position[2])
        force = total_contact_force(self.model, self.data, self.peg_id)
        proximity, _, _ = minimum_geom_proximity(
            self.model,
            self.data,
            self.peg_id,
            self.socket_wall_geom_ids,
            self.config.predictive_impact.distance_cutoff_m,
        )
        return xy_error, depth, force, proximity

    def _observation(self) -> NDArray[np.float32]:
        tip_position, _ = site_pose(self.model, self.data, self.tip_id)
        relative_position = self.hole_position - tip_position
        tip_velocity = site_jacobian(self.model, self.data, self.tip_id) @ self.data.qvel
        _, _, force, proximity = self._metrics()
        offset = self.action_filter.offset_m
        if offset is None:
            offset = np.zeros(3)
        observation = np.concatenate(
            (
                relative_position / np.array([0.012, 0.012, 0.080]),
                tip_velocity[:3] / 0.20,
                np.array(
                    [
                        force / self.config.force_safety.force_limit_n,
                        proximity / self.config.predictive_impact.distance_cutoff_m,
                        float(self.data.time) / self.rl.episode_duration_s,
                    ]
                ),
                offset / np.asarray(self.rl.max_position_offset_m),
            )
        )
        return np.clip(observation, -5.0, 5.0).astype(np.float32)

    def _info(
        self,
        *,
        success: bool,
        safety_violation: bool,
        xy_error_m: float,
        depth_m: float,
        force_n: float,
    ) -> dict[str, Any]:
        return {
            "is_success": success,
            "safety_violation": safety_violation,
            "time_s": float(self.data.time),
            "max_contact_force_n": self.maximum_force_n,
            "contact_force_n": force_n,
            "xy_error_m": xy_error_m,
            "insertion_depth_m": depth_m,
            "completion_time_s": self.completion_time_s,
            "residual_offset_m": self.action_filter.offset_m.copy().tolist()
            if self.action_filter.offset_m is not None
            else [0.0, 0.0, 0.0],
            **asdict(self.variation),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.variation = self._sample_variation(options)
        self._configure_episode()
        self.action_filter.reset()
        self.maximum_force_n = 0.0
        self.completion_time_s = None
        self.episode_done = False
        xy_error, depth, force, _ = self._metrics()
        self.previous_xy_error_m = xy_error
        self.previous_depth_m = depth
        return self._observation(), self._info(
            success=False,
            safety_violation=False,
            xy_error_m=xy_error,
            depth_m=depth,
            force_n=force,
        )

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.episode_done:
            raise RuntimeError("step() called after episode end; call reset()")
        residual = self.action_filter.command(
            np.asarray(action, dtype=float),
            self.rl.control_interval_s,
            enabled=float(self.data.time) >= self.config.peg_task.approach_time,
        )
        success = False
        safety_violation = False
        force = 0.0
        for _ in range(self.action_repeat):
            target_position, target_rotation = self._nominal_target(float(self.data.time))
            target_position += residual
            torque, _ = self.controller.torque(
                self.model,
                self.data,
                self.ee_id,
                target_position,
                target_rotation,
            )
            apply_torque(self.model, self.data, torque)
            mujoco.mj_step(self.model, self.data)
            xy_error, depth, force, _ = self._metrics()
            self.maximum_force_n = max(self.maximum_force_n, force)
            geometric_success = bool(
                xy_error <= self.config.peg_task.clearance
                and depth >= self.config.peg_task.insertion_depth
            )
            safety_violation = force >= self.config.force_safety.force_limit_n
            success = geometric_success and not safety_violation
            if success:
                self.completion_time_s = float(self.data.time)
            if success or safety_violation or self.data.time >= self.rl.episode_duration_s:
                break

        xy_progress = (self.previous_xy_error_m - xy_error) / self.config.peg_task.clearance
        depth_progress = (depth - self.previous_depth_m) / self.config.peg_task.insertion_depth
        force_ratio = force / self.config.force_safety.force_limit_n
        reward = (
            2.0 * xy_progress
            + 5.0 * depth_progress
            - 0.01 * min((xy_error / self.config.peg_task.clearance) ** 2, 9.0)
            - 0.02 * force_ratio * force_ratio
            - 0.002 * float(np.dot(action, action))
            - 0.001
        )
        if success:
            reward += self.rl.success_bonus
        if safety_violation:
            reward -= self.rl.safety_penalty
        self.previous_xy_error_m = xy_error
        self.previous_depth_m = depth
        terminated = success or safety_violation
        truncated = bool(not terminated and self.data.time >= self.rl.episode_duration_s)
        if truncated:
            remaining_depth = max(
                0.0,
                self.config.peg_task.insertion_depth - depth,
            )
            reward -= 5.0 * (
                xy_error / self.config.peg_task.clearance
                + remaining_depth / self.config.peg_task.insertion_depth
            )
        self.episode_done = terminated or truncated
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(
                success=success,
                safety_violation=safety_violation,
                xy_error_m=xy_error,
                depth_m=depth,
                force_n=force,
            ),
        )


def evaluate_residual_policy(
    config: ProjectConfig,
    policy: Policy,
    episodes: int,
    seed: int | None = None,
) -> list[ResidualEpisodeResult]:
    """Evaluate a policy on deterministic randomized episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = ResidualPegInsertionEnv(config)
    base_seed = config.residual_rl.seed if seed is None else seed
    results: list[ResidualEpisodeResult] = []
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=base_seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = np.asarray(policy(observation), dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            results.append(
                ResidualEpisodeResult(
                    episode=episode,
                    success=bool(info["is_success"]),
                    safety_violation=bool(info["safety_violation"]),
                    total_reward=total_reward,
                    steps=steps,
                    completion_time_s=info["completion_time_s"],
                    max_contact_force_n=float(info["max_contact_force_n"]),
                    final_xy_error_m=float(info["xy_error_m"]),
                    final_insertion_depth_m=float(info["insertion_depth_m"]),
                    lateral_offset_x=float(info["lateral_offset_x"]),
                    lateral_offset_y=float(info["lateral_offset_y"]),
                    friction_scale=float(info["friction_scale"]),
                    tool_mass_scale=float(info["tool_mass_scale"]),
                    final_residual_x_m=float(info["residual_offset_m"][0]),
                    final_residual_y_m=float(info["residual_offset_m"][1]),
                    final_residual_z_m=float(info["residual_offset_m"][2]),
                )
            )
    finally:
        env.close()
    return results


def zero_residual_policy(observation: NDArray[np.float32]) -> NDArray[np.float32]:
    del observation
    return np.zeros(3, dtype=np.float32)


def pose_servo_reference_policy(observation: NDArray[np.float32]) -> NDArray[np.float32]:
    """Privileged pose-servo reference used only to sanity-check environment solvability."""
    relative_xy_m = observation[:2].astype(float) * 0.012
    action = np.zeros(3, dtype=np.float32)
    action[:2] = np.clip(relative_xy_m / 0.008, -1.0, 1.0)
    return action


def collect_reference_dataset(
    config: ProjectConfig,
    episodes: int,
    seed: int | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Collect privileged pose-reference observations and actions for actor warm-starting."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = ResidualPegInsertionEnv(config)
    base_seed = config.residual_rl.seed if seed is None else seed
    observations: list[NDArray[np.float32]] = []
    actions: list[NDArray[np.float32]] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=base_seed + episode)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = pose_servo_reference_policy(observation)
                observations.append(observation.copy())
                actions.append(action.copy())
                observation, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def save_episode_results(results: list[ResidualEpisodeResult], path: Path) -> None:
    import csv
    import json

    path.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    if rows:
        with (path / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "episodes": len(results),
        "successes": sum(result.success for result in results),
        "success_rate": (
            sum(result.success for result in results) / len(results) if results else 0.0
        ),
        "safety_violations": sum(result.safety_violation for result in results),
        "mean_peak_force_n": (
            float(np.mean([result.max_contact_force_n for result in results])) if results else 0.0
        ),
        "mean_reward": (
            float(np.mean([result.total_reward for result in results])) if results else 0.0
        ),
    }
    with (path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
