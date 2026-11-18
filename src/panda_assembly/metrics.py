from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RunSummary:
    mode: str
    success: bool
    rms_position_error_m: float | None
    max_position_error_m: float | None
    max_contact_force_n: float
    completion_time_s: float | None
    rms_joint_error_rad: float | None = None
    max_joint_error_rad: float | None = None
    rms_orientation_error_rad: float | None = None
    max_orientation_error_rad: float | None = None
    final_xy_error_m: float | None = None
    final_insertion_depth_m: float | None = None
    controller: str | None = None
    outcome: str | None = None
    failure_type: str | None = None
    final_contact_force_n: float | None = None
    safety_triggered: bool = False
    safety_trigger_time_s: float | None = None
    safety_trigger_force_n: float | None = None
    retraction_distance_m: float | None = None
    safety_recovered: bool = False
    contact_recovery_triggered: bool = False
    jam_detected_time_s: float | None = None
    recovery_attempts: int = 0
    spiral_entry_detected: bool = False
    retry_success: bool = False


class RunLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, float | str | None]] = []

    def append(
        self,
        time: float,
        position_error: FloatArray | None,
        contact_force: float,
        qpos: FloatArray,
        joint_error: FloatArray | None = None,
        orientation_error: FloatArray | None = None,
        safety_state: str | None = None,
    ) -> None:
        row: dict[str, float | str | None] = {
            "time_s": float(time),
            "position_error_x_m": None,
            "position_error_y_m": None,
            "position_error_z_m": None,
            "position_error_norm_m": None,
            "joint_error_norm_rad": (
                None if joint_error is None else float(np.linalg.norm(joint_error))
            ),
            "orientation_error_norm_rad": (
                None if orientation_error is None else float(np.linalg.norm(orientation_error))
            ),
            "contact_force_n": float(contact_force),
            "safety_state": safety_state,
        }
        if position_error is not None:
            row.update(
                {
                    "position_error_x_m": float(position_error[0]),
                    "position_error_y_m": float(position_error[1]),
                    "position_error_z_m": float(position_error[2]),
                    "position_error_norm_m": float(np.linalg.norm(position_error)),
                }
            )
        row.update({f"q{i + 1}_rad": float(value) for i, value in enumerate(qpos[:7])})
        self.rows.append(row)

    def position_errors(self) -> FloatArray:
        return np.asarray(
            [
                row["position_error_norm_m"]
                for row in self.rows
                if row["position_error_norm_m"] is not None
            ],
            dtype=float,
        )

    def joint_errors(self) -> FloatArray:
        return np.asarray(
            [
                row["joint_error_norm_rad"]
                for row in self.rows
                if row["joint_error_norm_rad"] is not None
            ],
            dtype=float,
        )

    def orientation_errors(self) -> FloatArray:
        return np.asarray(
            [
                row["orientation_error_norm_rad"]
                for row in self.rows
                if row["orientation_error_norm_rad"] is not None
            ],
            dtype=float,
        )

    def max_contact_force(self) -> float:
        return max((float(row["contact_force_n"]) for row in self.rows), default=0.0)

    def final_contact_force(self) -> float:
        return float(self.rows[-1]["contact_force_n"]) if self.rows else 0.0

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def total_contact_force(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    total = 0.0
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        if geom_id not in (contact.geom1, contact.geom2):
            continue
        mujoco.mj_contactForce(model, data, index, force)
        total += float(np.linalg.norm(force[:3]))
    return total


def save_summary(summary: RunSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)


def save_plot(log: RunLog, path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    times = [row["time_s"] for row in log.rows]
    if log.position_errors().size:
        errors = [float(row["position_error_norm_m"]) * 1000.0 for row in log.rows]
        error_label = "Position error [mm]"
    else:
        errors = [float(row["joint_error_norm_rad"]) for row in log.rows]
        error_label = "Joint error norm [rad]"
    forces = [row["contact_force_n"] for row in log.rows]
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(times, errors, color="#2166ac")
    axes[0].set_ylabel(error_label)
    axes[0].grid(alpha=0.25)
    axes[1].plot(times, forces, color="#b2182b")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Peg contact force [N]")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True
