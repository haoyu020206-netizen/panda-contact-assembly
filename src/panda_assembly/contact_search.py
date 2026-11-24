from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from panda_assembly.pose_estimation import PoseEstimate

FloatArray = NDArray[np.float64]
Policy = Callable[[NDArray[np.float32]], NDArray[np.float32]]
EntryMode = Literal["legacy", "depth", "compensated"]


@dataclass(frozen=True)
class StallTriggerConfig:
    contact_duration_s: float
    minimum_progress_m: float
    retry_margin_s: float = 0.5

    def __post_init__(self) -> None:
        values = (self.contact_duration_s, self.minimum_progress_m, self.retry_margin_s)
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("stall-trigger settings must be positive and finite")


class SpiralResidualPolicy:
    """Add bounded unload-and-spiral recovery around a residual policy.

    The wrapper uses the residual environment's normalized force, episode phase, and
    hole-relative Z observation. It never changes impedance gains or actuator limits.
    """

    def __init__(
        self,
        base_policy: Policy,
        *,
        max_position_offset_m: FloatArray,
        episode_duration_s: float,
        force_scale_n: float,
        trigger_force_n: float = 5.0,
        retract_duration_s: float = 0.5,
        search_duration_s: float = 4.0,
        search_radius_m: float = 0.004,
        search_cycles: float = 3.0,
        entry_depth_m: float = 0.003,
        entry_mode: EntryMode = "legacy",
        max_pose_age_s: float = 0.16,
        stall_trigger: StallTriggerConfig | None = None,
    ) -> None:
        self.base_policy = base_policy
        self.max_position_offset_m = np.asarray(max_position_offset_m, dtype=float)
        if self.max_position_offset_m.shape != (3,) or np.any(self.max_position_offset_m <= 0.0):
            raise ValueError("max_position_offset_m must contain three positive values")
        positive_values = (
            episode_duration_s,
            force_scale_n,
            trigger_force_n,
            retract_duration_s,
            search_duration_s,
            search_radius_m,
            search_cycles,
            entry_depth_m,
            max_pose_age_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_values):
            raise ValueError("spiral recovery parameters must be positive and finite")
        if search_radius_m > float(np.min(self.max_position_offset_m[:2])):
            raise ValueError("search radius cannot exceed the lateral residual limit")
        self.episode_duration_s = float(episode_duration_s)
        self.force_scale_n = float(force_scale_n)
        self.trigger_force_n = float(trigger_force_n)
        self.retract_duration_s = float(retract_duration_s)
        self.search_duration_s = float(search_duration_s)
        self.search_radius_m = float(search_radius_m)
        self.search_cycles = float(search_cycles)
        self.entry_depth_m = float(entry_depth_m)
        if entry_mode not in ("legacy", "depth", "compensated"):
            raise ValueError(f"unknown entry mode: {entry_mode}")
        self.entry_mode = entry_mode
        self.max_pose_age_s = max_pose_age_s
        if stall_trigger is not None and entry_mode != "compensated":
            raise ValueError("stall recovery requires timestamped pose and tip kinematics")
        self.stall_trigger = stall_trigger
        self.reset_episode()

    def reset_episode(self) -> None:
        self._search_triggered = False
        self._entry_detected = False
        self._trigger_time_s = 0.0
        self._center_action = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(3, dtype=np.float32)
        self._tip_z_history: deque[tuple[float, float]] = deque()
        self._stall_start_s: float | None = None
        self._stall_tip_z_m = 0.0
        self.trigger_decision = "inactive"

    def episode_metrics(self) -> dict[str, bool]:
        return {
            "search_triggered": self._search_triggered,
            "entry_detected": self._entry_detected,
        }

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.entry_mode == "compensated":
            raise ValueError("compensated recovery requires timestamped pose and tip height")
        return self._action(observation)

    def act_with_pose(
        self,
        observation: NDArray[np.float32],
        pose: PoseEstimate,
        *,
        time_s: float,
        tip_z_m: float,
    ) -> NDArray[np.float32]:
        if self.entry_mode != "compensated":
            return self(observation)
        self._tip_z_history.append((time_s, tip_z_m))
        tolerance_s = 1e-9
        oldest_s = time_s - self.max_pose_age_s - tolerance_s
        while self._tip_z_history[0][0] < oldest_s:
            self._tip_z_history.popleft()
        depth_m = None
        if pose.age_s <= self.max_pose_age_s + tolerance_s:
            for sample_time_s, sample_z_m in self._tip_z_history:
                if abs(sample_time_s - pose.measurement_time_s) <= tolerance_s:
                    # The fixture is stationary. Only the robot tip moved since capture.
                    depth_m = float(pose.relative_position_m[2]) + sample_z_m - tip_z_m
                    break
        return self._action(observation, compensated_depth_m=depth_m, tip_z_m=tip_z_m)

    def _stall_ready(
        self,
        time_s: float,
        force_n: float,
        tip_z_m: float,
        depth_m: float | None,
    ) -> bool:
        config = self.stall_trigger
        assert config is not None
        if force_n < self.trigger_force_n or (
            depth_m is not None and depth_m >= self.entry_depth_m
        ):
            self._stall_start_s = None
            self.trigger_decision = "contact_or_depth_gate"
            return False
        if (
            self._stall_start_s is None
            or self._stall_tip_z_m - tip_z_m >= config.minimum_progress_m
        ):
            self._stall_start_s = time_s
            self._stall_tip_z_m = tip_z_m
        required_s = self.retract_duration_s + self.search_duration_s + config.retry_margin_s
        if self.episode_duration_s - time_s < required_s - 1e-6:
            self.trigger_decision = "insufficient_time"
        elif time_s - self._stall_start_s < config.contact_duration_s - 1e-6:
            self.trigger_decision = "collecting_contact_window"
        elif depth_m is None:
            self.trigger_decision = "stale_pose"
        else:
            self.trigger_decision = "stall_triggered"
        return self.trigger_decision == "stall_triggered"

    def _action(
        self,
        observation: NDArray[np.float32],
        *,
        compensated_depth_m: float | None = None,
        tip_z_m: float | None = None,
    ) -> NDArray[np.float32]:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (12,):
            raise ValueError(f"residual observation must have shape (12,), got {observation.shape}")
        base_action = np.asarray(self.base_policy(observation), dtype=np.float32)
        if base_action.shape != (3,):
            raise ValueError(f"base policy action must have shape (3,), got {base_action.shape}")
        base_action = np.clip(base_action, -1.0, 1.0)
        time_s = float(np.clip(observation[8], 0.0, 1.5) * self.episode_duration_s)
        force_n = max(0.0, float(observation[6]) * self.force_scale_n)
        raw_depth_m = float(observation[2]) * 0.080
        depth_m = compensated_depth_m if self.entry_mode == "compensated" else raw_depth_m
        legacy = self.entry_mode == "legacy"
        can_trigger = legacy or (depth_m is not None and depth_m < self.entry_depth_m)

        should_trigger = can_trigger and force_n >= self.trigger_force_n
        if not self._search_triggered and self.stall_trigger is not None:
            assert tip_z_m is not None
            should_trigger = self._stall_ready(time_s, force_n, tip_z_m, depth_m)
        if not self._search_triggered and should_trigger:
            self._search_triggered = True
            self._trigger_time_s = time_s
            self._center_action = base_action.copy()
            self._last_action = base_action.copy()

        if not self._search_triggered:
            return base_action
        elapsed_s = max(0.0, time_s - self._trigger_time_s)
        if legacy:
            entered = raw_depth_m <= -self.entry_depth_m
        else:
            entered = (
                elapsed_s >= self.retract_duration_s
                and depth_m is not None
                and depth_m >= self.entry_depth_m
            )
        if entered:
            self._entry_detected = True
        if self._entry_detected:
            return self._last_action.copy()

        action = self._center_action.copy()
        if elapsed_s < self.retract_duration_s:
            action[2] = 1.0
        else:
            search_elapsed_s = elapsed_s - self.retract_duration_s
            progress = search_elapsed_s / self.search_duration_s
            radial_progress = min(progress, 1.0)
            radius_m = self.search_radius_m * radial_progress
            angle_rad = 2.0 * np.pi * self.search_cycles * progress
            correction_m = radius_m * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            action[:2] += correction_m / self.max_position_offset_m[:2]
            action[2] = max(0.0, 1.0 - radial_progress)
        self._last_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        return self._last_action.copy()
