from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


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
        progress = float(np.clip(elapsed / self.retract_duration_s, 0.0, 1.0))
        blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
        target = self.trigger_position.copy()
        target[2] += blend * self.retract_distance_m
        return target, self.trigger_rotation.copy(), state


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
) -> tuple[str, str | None]:
    """Return an outcome and its primary failure type for aggregate reporting."""
    if safety_triggered:
        outcome = "safety_abort_recovered" if safety_recovered else "safety_abort"
        return outcome, "excessive_force"
    if geometric_success:
        if max_contact_force_n > force_limit_n:
            return "unsafe_completion", "excessive_force"
        return "success", None
    if max_contact_force_n > force_limit_n:
        return "excessive_force", "excessive_force"
    if final_contact_force_n > 1.0:
        return "jammed", "jammed"
    if final_xy_error_m > clearance_m:
        return "lateral_misalignment", "lateral_misalignment"
    if final_insertion_depth_m < insertion_depth_m:
        return "insufficient_depth", "insufficient_depth"
    return "incomplete", "unknown"
