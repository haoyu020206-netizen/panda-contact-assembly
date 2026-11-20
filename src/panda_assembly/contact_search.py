from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Policy = Callable[[NDArray[np.float32]], NDArray[np.float32]]


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
        self.reset_episode()

    def reset_episode(self) -> None:
        self._search_triggered = False
        self._entry_detected = False
        self._trigger_time_s = 0.0
        self._center_action = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(3, dtype=np.float32)

    def episode_metrics(self) -> dict[str, bool]:
        return {
            "search_triggered": self._search_triggered,
            "entry_detected": self._entry_detected,
        }

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (12,):
            raise ValueError(f"residual observation must have shape (12,), got {observation.shape}")
        base_action = np.asarray(self.base_policy(observation), dtype=np.float32)
        if base_action.shape != (3,):
            raise ValueError(f"base policy action must have shape (3,), got {base_action.shape}")
        base_action = np.clip(base_action, -1.0, 1.0)
        time_s = float(np.clip(observation[8], 0.0, 1.5) * self.episode_duration_s)
        force_n = max(0.0, float(observation[6]) * self.force_scale_n)
        relative_z_m = float(observation[2]) * 0.080

        if not self._search_triggered and force_n >= self.trigger_force_n:
            self._search_triggered = True
            self._trigger_time_s = time_s
            self._center_action = base_action.copy()
            self._last_action = base_action.copy()

        if not self._search_triggered:
            return base_action
        if not self._entry_detected and relative_z_m <= -self.entry_depth_m:
            self._entry_detected = True
        if self._entry_detected:
            return self._last_action.copy()

        elapsed_s = max(0.0, time_s - self._trigger_time_s)
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
