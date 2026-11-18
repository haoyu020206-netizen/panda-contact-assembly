from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _minimum_jerk(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


@dataclass
class ForceSafetySupervisor:
    """Latch excessive contact and replace insertion with a smooth upward retreat."""

    force_limit_n: float
    trigger_duration_s: float
    hold_time_s: float
    retract_distance_m: float
    retract_duration_s: float
    over_limit_since_s: float | None = None
    trigger_time_s: float | None = None
    trigger_force_n: float | None = None
    trigger_position: FloatArray | None = None
    trigger_rotation: FloatArray | None = None

    @property
    def triggered(self) -> bool:
        return self.trigger_time_s is not None

    def update(
        self,
        sim_time: float,
        contact_force_n: float,
        current_position: FloatArray,
        current_rotation: FloatArray,
    ) -> bool:
        """Update the force monitor and return True only on the triggering sample."""
        if self.triggered:
            return False
        if contact_force_n < self.force_limit_n:
            self.over_limit_since_s = None
            return False
        if self.over_limit_since_s is None:
            self.over_limit_since_s = sim_time
        if sim_time - self.over_limit_since_s + 1e-12 < self.trigger_duration_s:
            return False
        self.trigger_time_s = sim_time
        self.trigger_force_n = contact_force_n
        self.trigger_position = current_position.copy()
        self.trigger_rotation = current_rotation.copy()
        return True

    def state(self, sim_time: float) -> str:
        if self.trigger_time_s is None:
            return "monitoring"
        elapsed = max(0.0, sim_time - self.trigger_time_s)
        if elapsed <= self.hold_time_s:
            return "stop"
        if elapsed < self.hold_time_s + self.retract_duration_s:
            return "retract"
        return "retracted"

    def command(
        self,
        sim_time: float,
        nominal_position: FloatArray,
        nominal_rotation: FloatArray,
    ) -> tuple[FloatArray, FloatArray, str]:
        state = self.state(sim_time)
        if not self.triggered:
            return nominal_position, nominal_rotation, state
        assert self.trigger_time_s is not None
        assert self.trigger_position is not None
        assert self.trigger_rotation is not None
        if state == "stop":
            return self.trigger_position.copy(), self.trigger_rotation.copy(), state
        elapsed = sim_time - self.trigger_time_s - self.hold_time_s
        blend = _minimum_jerk(elapsed / self.retract_duration_s)
        target = self.trigger_position.copy()
        target[2] += blend * self.retract_distance_m
        return target, self.trigger_rotation.copy(), state


@dataclass
class InsertionSpeedLimiter:
    """Rate-limit the axial command near the socket and during light contact."""

    timestep_s: float
    precontact_distance_m: float
    slow_zone_depth_m: float
    free_speed_m_s: float
    precontact_speed_m_s: float
    contact_slow_force_n: float
    contact_speed_m_s: float
    commanded_z_m: float | None = None

    def command(
        self,
        nominal_position: FloatArray,
        tip_z_m: float,
        hole_z_m: float,
        contact_force_n: float,
        insertion_active: bool,
        speed_cap_m_s: float | None = None,
    ) -> tuple[FloatArray, str]:
        if not insertion_active:
            self.commanded_z_m = float(nominal_position[2])
            return nominal_position, "free"
        if self.commanded_z_m is None:
            self.commanded_z_m = float(nominal_position[2])

        gap_m = tip_z_m - hole_z_m
        speed = self.free_speed_m_s
        state = "free"
        if -self.slow_zone_depth_m <= gap_m <= self.precontact_distance_m:
            speed = self.precontact_speed_m_s
            state = "precontact_slow"
        if contact_force_n >= self.contact_slow_force_n:
            speed = self.contact_speed_m_s
            state = "contact_slow"
        if speed_cap_m_s is not None and speed_cap_m_s < speed:
            speed = speed_cap_m_s
            state = "proximity_probe"

        self.commanded_z_m = max(
            float(nominal_position[2]),
            self.commanded_z_m - speed * self.timestep_s,
        )
        target = nominal_position.copy()
        target[2] = self.commanded_z_m
        return target, state


@dataclass
class PredictiveImpactGuard:
    """Use proximity and force slope to trigger recovery before a large impact."""

    timestep_s: float
    slowdown_distance_m: float
    trigger_distance_m: float
    minimum_closing_speed_m_s: float
    probe_speed_m_s: float
    force_rate_limit_n_s: float
    force_rate_min_force_n: float
    previous_distance_m: float | None = None
    previous_force_n: float = 0.0
    last_distance_m: float | None = None
    last_closing_speed_m_s: float = 0.0
    last_force_rate_n_s: float = 0.0
    minimum_distance_m: float | None = None
    maximum_force_rate_n_s: float = 0.0
    trigger_time_s: float | None = None
    trigger_reason: str | None = None
    triggered_at_distance_m: float | None = None
    triggered_at_closing_speed_m_s: float | None = None
    triggered_at_force_rate_n_s: float | None = None

    @property
    def triggered(self) -> bool:
        return self.trigger_time_s is not None

    @property
    def slowing(self) -> bool:
        return self.last_distance_m is not None and self.last_distance_m <= self.slowdown_distance_m

    @property
    def speed_cap_m_s(self) -> float | None:
        return self.probe_speed_m_s if self.slowing else None

    @property
    def state_name(self) -> str:
        if self.triggered:
            return f"triggered_{self.trigger_reason}"
        if self.slowing:
            return "proximity_slow"
        return "monitoring"

    def update(
        self,
        sim_time: float,
        proximity_distance_m: float,
        contact_force_n: float,
        active: bool,
    ) -> bool:
        closing_speed = 0.0
        if self.previous_distance_m is not None:
            closing_speed = max(
                0.0,
                (self.previous_distance_m - proximity_distance_m) / self.timestep_s,
            )
        force_rate = max(0.0, (contact_force_n - self.previous_force_n) / self.timestep_s)
        self.previous_distance_m = proximity_distance_m
        self.previous_force_n = contact_force_n
        self.last_distance_m = proximity_distance_m
        self.last_closing_speed_m_s = closing_speed
        self.last_force_rate_n_s = force_rate

        if active:
            if self.minimum_distance_m is None:
                self.minimum_distance_m = proximity_distance_m
            else:
                self.minimum_distance_m = min(self.minimum_distance_m, proximity_distance_m)
            self.maximum_force_rate_n_s = max(self.maximum_force_rate_n_s, force_rate)
        if not active or self.triggered:
            return False

        reason = None
        if (
            proximity_distance_m <= self.trigger_distance_m
            and closing_speed >= self.minimum_closing_speed_m_s
        ):
            reason = "proximity"
        elif (
            contact_force_n >= self.force_rate_min_force_n
            and force_rate >= self.force_rate_limit_n_s
        ):
            reason = "force_rate"
        if reason is None:
            return False

        self.trigger_time_s = sim_time
        self.trigger_reason = reason
        self.triggered_at_distance_m = proximity_distance_m
        self.triggered_at_closing_speed_m_s = closing_speed
        self.triggered_at_force_rate_n_s = force_rate
        return True


@dataclass
class LateralCommandLimiter:
    """Rate-limit recovery XY commands without changing their geometric path."""

    timestep_s: float
    speed_limit_m_s: float
    commanded_xy_m: FloatArray | None = None

    def command(self, nominal_position: FloatArray, enabled: bool) -> tuple[FloatArray, str]:
        if self.commanded_xy_m is None:
            self.commanded_xy_m = nominal_position[:2].copy()
        if not enabled:
            self.commanded_xy_m = nominal_position[:2].copy()
            return nominal_position, "free"

        delta = nominal_position[:2] - self.commanded_xy_m
        distance = float(np.linalg.norm(delta))
        maximum_step = self.speed_limit_m_s * self.timestep_s
        state = "tracking"
        if distance > maximum_step:
            self.commanded_xy_m += delta * (maximum_step / distance)
            state = "limited"
        else:
            self.commanded_xy_m = nominal_position[:2].copy()
        target = nominal_position.copy()
        target[:2] = self.commanded_xy_m
        return target, state


@dataclass
class SpiralRecoverySupervisor:
    """Detect a low-force jam, retract locally, spiral, and retry insertion."""

    jam_force_n: float
    jam_timeout_s: float
    jam_progress_m: float
    retract_distance_m: float
    retract_duration_s: float
    spiral_radius_m: float
    spiral_cycles: int
    spiral_duration_s: float
    spiral_descent_m: float
    entry_depth_m: float
    retry_insert_duration_s: float
    max_retries: int
    search_center: FloatArray
    final_target_z_m: float
    state_name: str = "monitoring"
    attempts: int = 0
    jam_window_start_s: float | None = None
    jam_reference_depth_m: float | None = None
    trigger_time_s: float | None = None
    trigger_position: FloatArray | None = None
    trigger_rotation: FloatArray | None = None
    search_start_s: float | None = None
    entry_time_s: float | None = None
    entry_position: FloatArray | None = None
    retry_start_s: float | None = None
    retry_success: bool = False

    @property
    def triggered(self) -> bool:
        return self.attempts > 0

    @property
    def entry_detected(self) -> bool:
        return self.entry_time_s is not None

    def _begin_recovery(
        self,
        sim_time: float,
        current_position: FloatArray,
        current_rotation: FloatArray,
    ) -> None:
        self.attempts += 1
        self.state_name = "local_retract"
        self.trigger_time_s = sim_time
        self.trigger_position = current_position.copy()
        self.trigger_rotation = current_rotation.copy()
        self.jam_window_start_s = None
        self.jam_reference_depth_m = None

    def trigger_recovery(
        self,
        sim_time: float,
        current_position: FloatArray,
        current_rotation: FloatArray,
    ) -> bool:
        """Begin recovery from an external predictive trigger if a retry is available."""
        if self.state_name != "monitoring" or self.attempts >= self.max_retries:
            return False
        self._begin_recovery(sim_time, current_position, current_rotation)
        return True

    def update(
        self,
        sim_time: float,
        contact_force_n: float,
        insertion_depth_m: float,
        task_complete: bool,
        current_position: FloatArray,
        current_rotation: FloatArray,
    ) -> bool:
        """Advance the recovery state and return True on a new jam trigger."""
        if self.state_name == "monitoring":
            if task_complete or contact_force_n < self.jam_force_n:
                self.jam_window_start_s = None
                self.jam_reference_depth_m = None
                return False
            if self.jam_window_start_s is None or self.jam_reference_depth_m is None:
                self.jam_window_start_s = sim_time
                self.jam_reference_depth_m = insertion_depth_m
                return False
            if insertion_depth_m >= self.jam_reference_depth_m + self.jam_progress_m:
                self.jam_window_start_s = sim_time
                self.jam_reference_depth_m = insertion_depth_m
                return False
            if (
                sim_time - self.jam_window_start_s >= self.jam_timeout_s
                and self.attempts < self.max_retries
            ):
                return self.trigger_recovery(sim_time, current_position, current_rotation)
            return False

        if self.state_name == "local_retract":
            assert self.trigger_time_s is not None
            if sim_time - self.trigger_time_s >= self.retract_duration_s:
                self.state_name = "spiral_search"
                self.search_start_s = sim_time
            return False

        if self.state_name == "spiral_search":
            assert self.search_start_s is not None
            search_elapsed = sim_time - self.search_start_s
            if insertion_depth_m >= self.entry_depth_m:
                self.state_name = "retry_insert"
                self.entry_time_s = sim_time
                self.entry_position = current_position.copy()
                self.retry_start_s = sim_time
            elif search_elapsed >= self.spiral_duration_s:
                self.state_name = "recovery_failed"
            return False

        if self.state_name == "retry_insert":
            assert self.retry_start_s is not None
            if task_complete:
                self.retry_success = True
                self.state_name = "completed"
            elif sim_time - self.retry_start_s > self.retry_insert_duration_s + 0.5:
                self.state_name = "recovery_failed"
        return False

    def command(
        self,
        sim_time: float,
        nominal_position: FloatArray,
        nominal_rotation: FloatArray,
    ) -> tuple[FloatArray, FloatArray, str]:
        if self.state_name == "monitoring":
            return nominal_position, nominal_rotation, self.state_name

        if self.state_name == "completed":
            assert self.entry_position is not None
            assert self.trigger_rotation is not None
            target = self.entry_position.copy()
            target[2] = self.final_target_z_m
            return target, self.trigger_rotation.copy(), self.state_name

        assert self.trigger_position is not None
        assert self.trigger_rotation is not None
        target = self.trigger_position.copy()
        if self.state_name == "local_retract":
            assert self.trigger_time_s is not None
            blend = _minimum_jerk((sim_time - self.trigger_time_s) / self.retract_duration_s)
            target[2] += blend * self.retract_distance_m
            return target, self.trigger_rotation.copy(), self.state_name

        if self.state_name == "spiral_search":
            assert self.search_start_s is not None
            progress = float(
                np.clip((sim_time - self.search_start_s) / self.spiral_duration_s, 0.0, 1.0)
            )
            radius = self.spiral_radius_m * progress
            angle = 2.0 * np.pi * self.spiral_cycles * progress
            target[0] = self.search_center[0] + radius * np.cos(angle)
            target[1] = self.search_center[1] + radius * np.sin(angle)
            if progress <= 0.25:
                target[2] += self.retract_distance_m * (1.0 - _minimum_jerk(progress / 0.25))
            else:
                target[2] -= self.spiral_descent_m * _minimum_jerk((progress - 0.25) / 0.75)
            return target, self.trigger_rotation.copy(), self.state_name

        if self.state_name == "retry_insert":
            assert self.retry_start_s is not None
            assert self.entry_position is not None
            target = self.entry_position.copy()
            blend = _minimum_jerk((sim_time - self.retry_start_s) / self.retry_insert_duration_s)
            target[2] = self.entry_position[2] + blend * (
                self.final_target_z_m - self.entry_position[2]
            )
            return target, self.trigger_rotation.copy(), self.state_name

        target[2] += self.retract_distance_m
        return target, self.trigger_rotation.copy(), self.state_name


def classify_peg_outcome(
    *,
    geometric_success: bool,
    max_contact_force_n: float,
    final_contact_force_n: float,
    final_xy_error_m: float,
    final_insertion_depth_m: float,
    clearance_m: float,
    insertion_depth_m: float,
    force_limit_n: float,
    safety_triggered: bool,
    safety_recovered: bool,
    contact_recovery_triggered: bool = False,
    retry_success: bool = False,
) -> tuple[str, str | None]:
    """Return an outcome and its primary failure type for aggregate reporting."""
    if safety_triggered:
        outcome = "safety_abort_recovered" if safety_recovered else "safety_abort"
        return outcome, "excessive_force"
    if geometric_success:
        if retry_success:
            return "success_after_recovery", None
        if max_contact_force_n > force_limit_n:
            return "unsafe_completion", "excessive_force"
        return "success", None
    if contact_recovery_triggered:
        return "recovery_failed", "jammed"
    if max_contact_force_n > force_limit_n:
        return "excessive_force", "excessive_force"
    if final_contact_force_n > 1.0:
        return "jammed", "jammed"
    if final_xy_error_m > clearance_m:
        return "lateral_misalignment", "lateral_misalignment"
    if final_insertion_depth_m < insertion_depth_m:
        return "insufficient_depth", "insufficient_depth"
    return "incomplete", "unknown"
