from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from panda_assembly.config import ProjectConfig
from panda_assembly.controllers import (
    CartesianImpedanceController,
    CartesianPositionController,
    JointPDController,
    apply_torque,
    site_pose,
)
from panda_assembly.math_utils import interpolate_rotation, smoothstep, vertical_tool_frame
from panda_assembly.metrics import (
    RunLog,
    RunSummary,
    save_plot,
    save_summary,
    total_contact_force,
)
from panda_assembly.model import load_model, reset_to_home
from panda_assembly.trajectories import quintic_joint_trajectory


@dataclass(frozen=True)
class Variation:
    lateral_offset_x: float = 0.0
    lateral_offset_y: float = 0.0
    friction_scale: float = 1.0
    tool_mass_scale: float = 1.0


ROBUSTNESS_SCENARIOS = (
    ("nominal", Variation()),
    ("pose_x_plus_2mm", Variation(lateral_offset_x=0.002)),
    ("pose_y_minus_2mm", Variation(lateral_offset_y=-0.002)),
    ("friction_low", Variation(friction_scale=0.6)),
    ("friction_high", Variation(friction_scale=1.4)),
    ("tool_light", Variation(tool_mass_scale=0.8)),
    ("tool_heavy", Variation(tool_mass_scale=1.2)),
    (
        "combined",
        Variation(lateral_offset_x=0.0055, friction_scale=1.4, tool_mass_scale=1.2),
    ),
)


def _site_offset_in_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin_site_id: int,
    target_site_id: int,
) -> np.ndarray:
    """Express the origin-to-target site offset in the origin site's frame."""
    origin_position, origin_rotation = site_pose(model, data, origin_site_id)
    target_position, _ = site_pose(model, data, target_site_id)
    return origin_rotation.T @ (target_position - origin_position)


def _cartesian_controller(
    config: ProjectConfig, rest_posture: np.ndarray
) -> CartesianImpedanceController:
    values = config.cartesian_controller
    return CartesianImpedanceController(
        position_stiffness=np.asarray(values.position_stiffness),
        position_damping=np.asarray(values.position_damping),
        orientation_stiffness=np.asarray(values.orientation_stiffness),
        orientation_damping=np.asarray(values.orientation_damping),
        nullspace_stiffness=values.nullspace_stiffness,
        nullspace_damping=values.nullspace_damping,
        damped_inverse_ratio=values.damping_ratio,
        rest_posture=rest_posture.copy(),
    )


def _cartesian_position_controller(
    config: ProjectConfig, initial_posture: np.ndarray
) -> CartesianPositionController:
    values = config.cartesian_position_controller
    return CartesianPositionController(
        joint_kp=np.asarray(values.joint_kp),
        joint_kd=np.asarray(values.joint_kd),
        translation_gain=values.translation_gain,
        rotation_gain=values.rotation_gain,
        damped_inverse_ratio=values.damping_ratio,
        max_joint_velocity=values.max_joint_velocity,
        desired_joint_position=initial_posture.copy(),
    )


def _task_controller(
    config: ProjectConfig,
    initial_posture: np.ndarray,
    controller_kind: str,
) -> CartesianImpedanceController | CartesianPositionController:
    if controller_kind == "impedance":
        return _cartesian_controller(config, initial_posture)
    if controller_kind == "position":
        return _cartesian_position_controller(config, initial_posture)
    raise ValueError("controller_kind must be either 'impedance' or 'position'")


def _run_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
    render_hz: float,
    callback: Callable[[float], None],
    viewer: bool,
) -> None:
    passive_viewer = None
    if viewer:
        from mujoco import viewer as mujoco_viewer

        passive_viewer = mujoco_viewer.launch_passive(model, data)
    next_render = 0.0
    try:
        while data.time < duration:
            wall_start = time.perf_counter()
            callback(float(data.time))
            mujoco.mj_step(model, data)
            if passive_viewer is not None and data.time >= next_render:
                passive_viewer.sync()
                next_render += 1.0 / render_hz
                remaining = model.opt.timestep - (time.perf_counter() - wall_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if passive_viewer is not None:
            passive_viewer.close()


def _output_directory(config: ProjectConfig, mode: str, output: Path | None) -> Path:
    if output is not None:
        return output
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(config.output.root) / f"{mode}_{stamp}"


def _finish_run(
    config: ProjectConfig,
    mode: str,
    log: RunLog,
    success: bool,
    completion_time: float | None,
    output: Path | None,
    final_xy_error: float | None = None,
    final_depth: float | None = None,
    controller: str | None = None,
) -> tuple[RunSummary, Path]:
    position_errors = log.position_errors()
    joint_errors = log.joint_errors()
    orientation_errors = log.orientation_errors()
    summary = RunSummary(
        mode=mode,
        success=success,
        rms_position_error_m=(
            float(np.sqrt(np.mean(position_errors**2))) if position_errors.size else None
        ),
        max_position_error_m=(float(np.max(position_errors)) if position_errors.size else None),
        max_contact_force_n=log.max_contact_force(),
        completion_time_s=completion_time,
        rms_joint_error_rad=(
            float(np.sqrt(np.mean(joint_errors**2))) if joint_errors.size else None
        ),
        max_joint_error_rad=float(np.max(joint_errors)) if joint_errors.size else None,
        rms_orientation_error_rad=(
            float(np.sqrt(np.mean(orientation_errors**2))) if orientation_errors.size else None
        ),
        max_orientation_error_rad=(
            float(np.max(orientation_errors)) if orientation_errors.size else None
        ),
        final_xy_error_m=final_xy_error,
        final_insertion_depth_m=final_depth,
        controller=controller,
    )
    directory = _output_directory(config, mode, output)
    directory.mkdir(parents=True, exist_ok=True)
    if config.output.save_csv:
        log.save_csv(directory / "timeseries.csv")
    save_summary(summary, directory / "summary.json")
    if config.output.save_plot:
        save_plot(log, directory / "metrics.png")
    return summary, directory


def run_joint(
    config: ProjectConfig,
    viewer: bool = False,
    output: Path | None = None,
) -> tuple[RunSummary, Path]:
    model = load_model(timestep=config.simulation.timestep, variant=config.model.variant)
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    start = data.qpos[:7].copy()
    goal = start + np.asarray(config.joint_controller.target_offset)
    controller = JointPDController(
        kp=np.asarray(config.joint_controller.kp),
        kd=np.asarray(config.joint_controller.kd),
    )
    log = RunLog()

    def step(sim_time: float) -> None:
        target = quintic_joint_trajectory(
            start, goal, sim_time, config.joint_controller.move_duration
        )
        torque = controller.torque(data, target.position, target.velocity)
        apply_torque(model, data, torque)
        joint_error = target.position - data.qpos[:7]
        log.append(sim_time, None, 0.0, data.qpos, joint_error=joint_error)

    _run_loop(
        model,
        data,
        config.simulation.duration,
        config.simulation.render_hz,
        step,
        viewer,
    )
    final_error = float(np.linalg.norm(goal - data.qpos[:7]))
    return _finish_run(
        config,
        "joint",
        log,
        success=final_error < 0.03,
        completion_time=config.simulation.duration,
        output=output,
        controller="joint_pd",
    )


def run_cartesian(
    config: ProjectConfig,
    viewer: bool = False,
    output: Path | None = None,
    controller_kind: str = "impedance",
) -> tuple[RunSummary, Path]:
    model = load_model(timestep=config.simulation.timestep, variant=config.model.variant)
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    start_position, desired_rotation = site_pose(model, data, ee_id)
    goal = start_position + np.asarray(config.cartesian_controller.target_offset)
    controller = _task_controller(config, data.qpos[:7], controller_kind)
    log = RunLog()

    def step(sim_time: float) -> None:
        blend = smoothstep(sim_time / 3.0)
        desired_position = start_position + blend * (goal - start_position)
        torque, error = controller.torque(model, data, ee_id, desired_position, desired_rotation)
        apply_torque(model, data, torque)
        log.append(
            sim_time,
            error[:3],
            0.0,
            data.qpos,
            orientation_error=error[3:],
        )

    _run_loop(
        model,
        data,
        config.simulation.duration,
        config.simulation.render_hz,
        step,
        viewer,
    )
    final_position, _ = site_pose(model, data, ee_id)
    final_error = float(np.linalg.norm(goal - final_position))
    return _finish_run(
        config,
        "cartesian",
        log,
        success=final_error < 0.020,
        completion_time=config.simulation.duration,
        output=output,
        controller=controller_kind,
    )


def _configure_variation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    variation: Variation,
    nominal_friction: float,
) -> tuple[int, int, int]:
    fixture_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "socket_fixture")
    mocap_id = model.body_mocapid[fixture_body]
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    peg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
    tool_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")

    model.body_mass[tool_body] *= variation.tool_mass_scale
    for name in ("socket_x_pos", "socket_x_neg", "socket_y_pos", "socket_y_neg"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        model.geom_friction[geom_id, 0] = nominal_friction * variation.friction_scale
    mujoco.mj_setConst(model, data)
    # mj_setConst recomputes model-derived quantities and resets data to qpos0.
    # Restore the named home posture before positioning the movable fixture.
    reset_to_home(model, data)
    position, _ = site_pose(model, data, ee_id)
    data.mocap_pos[mocap_id] = np.array(
        [
            position[0] + variation.lateral_offset_x,
            position[1] + variation.lateral_offset_y,
            0.30,
        ]
    )
    mujoco.mj_forward(model, data)
    hole_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hole_target")
    return ee_id, peg_id, hole_site


def run_peg(
    config: ProjectConfig,
    viewer: bool = False,
    output: Path | None = None,
    variation: Variation | None = None,
    controller_kind: str = "impedance",
) -> tuple[RunSummary, Path]:
    variation = variation or Variation()
    model = load_model(timestep=config.simulation.timestep, variant=config.model.variant)
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    ee_id, peg_id, hole_id = _configure_variation(
        model, data, variation, config.peg_task.nominal_friction
    )
    start_position, start_rotation = site_pose(model, data, ee_id)
    hole_position, _ = site_pose(model, data, hole_id)
    commanded_hole_position = hole_position.copy()
    commanded_hole_position[0] -= variation.lateral_offset_x
    commanded_hole_position[1] -= variation.lateral_offset_y
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
    vertical_rotation = vertical_tool_frame(start_rotation)
    tip_offset_local = _site_offset_in_frame(model, data, ee_id, tip_id)
    vertical_tip_offset = vertical_rotation @ tip_offset_local
    controller = _task_controller(config, data.qpos[:7], controller_kind)
    peg = config.peg_task
    log = RunLog()
    completion_time: float | None = None

    aligned = np.array(
        [
            commanded_hole_position[0] - vertical_tip_offset[0],
            commanded_hole_position[1] - vertical_tip_offset[1],
            start_position[2],
        ]
    )
    final_target = aligned.copy()
    final_target[2] = commanded_hole_position[2] - peg.target_depth - vertical_tip_offset[2]

    def desired_position(sim_time: float) -> np.ndarray:
        if sim_time <= peg.approach_time:
            return start_position
        if sim_time <= peg.align_time:
            progress = (sim_time - peg.approach_time) / (peg.align_time - peg.approach_time)
            return start_position + smoothstep(progress) * (aligned - start_position)
        progress = (sim_time - peg.align_time) / (peg.insert_time - peg.align_time)
        return aligned + smoothstep(progress) * (final_target - aligned)

    def desired_rotation(sim_time: float) -> np.ndarray:
        if sim_time <= peg.approach_time:
            return start_rotation
        progress = (sim_time - peg.approach_time) / (peg.align_time - peg.approach_time)
        return interpolate_rotation(start_rotation, vertical_rotation, progress)

    def step(sim_time: float) -> None:
        nonlocal completion_time
        target = desired_position(sim_time)
        target_rotation = desired_rotation(sim_time)
        torque, error = controller.torque(model, data, ee_id, target, target_rotation)
        apply_torque(model, data, torque)
        force = total_contact_force(model, data, peg_id)
        log.append(
            sim_time,
            error[:3],
            force,
            data.qpos,
            orientation_error=error[3:],
        )

        tip_position, _ = site_pose(model, data, tip_id)
        xy_error = float(np.linalg.norm(tip_position[:2] - hole_position[:2]))
        depth = float(hole_position[2] - tip_position[2])
        if completion_time is None and xy_error <= peg.clearance and depth >= peg.insertion_depth:
            completion_time = sim_time

    duration = max(config.simulation.duration, peg.settle_time)
    _run_loop(model, data, duration, config.simulation.render_hz, step, viewer)
    final_tip, _ = site_pose(model, data, tip_id)
    final_xy_error = float(np.linalg.norm(final_tip[:2] - hole_position[:2]))
    final_depth = float(hole_position[2] - final_tip[2])
    return _finish_run(
        config,
        "peg",
        log,
        success=completion_time is not None,
        completion_time=completion_time,
        output=output,
        final_xy_error=final_xy_error,
        final_depth=final_depth,
        controller=controller_kind,
    )


def run_robustness_batch(
    config: ProjectConfig,
    output: Path | None = None,
) -> Path:
    directory = _output_directory(config, "batch", output)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, (scenario, variation) in enumerate(ROBUSTNESS_SCENARIOS):
        summary, _ = run_peg(
            config,
            viewer=False,
            output=directory / f"trial_{index:02d}",
            variation=variation,
        )
        rows.append({"scenario": scenario, **asdict(variation), **asdict(summary)})

    with (directory / "batch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(bool(row["success"]) for row in rows)
    aggregate = {
        "trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_max_contact_force_n": float(
            np.mean([float(row["max_contact_force_n"]) for row in rows])
        ),
    }
    with (directory / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    return directory


def _save_controller_comparison_plot(directory: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    colors = {"impedance": "#2166ac", "position": "#b2182b"}
    for controller_kind, color in colors.items():
        with (directory / controller_kind / "timeseries.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        times = [float(row["time_s"]) for row in rows]
        errors = [1000.0 * float(row["position_error_norm_m"]) for row in rows]
        forces = [float(row["contact_force_n"]) for row in rows]
        axes[0].plot(times, errors, color=color, label=controller_kind)
        axes[1].plot(times, forces, color=color, label=controller_kind)
    axes[0].set_ylabel("Position error [mm]")
    axes[1].set_ylabel("Peg contact force [N]")
    axes[1].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "comparison.png", dpi=160)
    plt.close(figure)
    return True


def run_controller_comparison(config: ProjectConfig, output: Path | None = None) -> Path:
    """Run the same nominal insertion with position and impedance control."""
    directory = _output_directory(config, "comparison", output)
    directory.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, RunSummary] = {}
    for controller_kind in ("impedance", "position"):
        summary, _ = run_peg(
            config,
            viewer=False,
            output=directory / controller_kind,
            controller_kind=controller_kind,
        )
        summaries[controller_kind] = summary

    rows = [asdict(summaries[name]) for name in ("impedance", "position")]
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    impedance_force = summaries["impedance"].max_contact_force_n
    position_force = summaries["position"].max_contact_force_n
    comparison = {
        "model_variant": config.model.variant,
        "task": "nominal_peg_insertion",
        "controllers": {name: asdict(summary) for name, summary in summaries.items()},
        "position_minus_impedance_peak_force_n": position_force - impedance_force,
        "position_over_impedance_peak_force_ratio": (
            position_force / impedance_force if impedance_force > 0.0 else None
        ),
    }
    with (directory / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    if config.output.save_plot and config.output.save_csv:
        _save_controller_comparison_plot(directory)
    return directory


def _aggregate_controller_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    successes = sum(bool(row["success"]) for row in rows)
    forces = [float(row["max_contact_force_n"]) for row in rows]
    completion_times = [
        float(row["completion_time_s"]) for row in rows if row["completion_time_s"] is not None
    ]
    return {
        "trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_peak_contact_force_n": float(np.mean(forces)),
        "max_peak_contact_force_n": float(np.max(forces)),
        "mean_completion_time_s_successes": (
            float(np.mean(completion_times)) if completion_times else None
        ),
    }


def _save_robustness_comparison_plot(rows: list[dict[str, object]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    scenarios = [name for name, _ in ROBUSTNESS_SCENARIOS]
    indices = np.arange(len(scenarios))
    width = 0.38
    colors = {"impedance": "#2166ac", "position": "#b2182b"}
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
    for offset, controller_kind in ((-width / 2, "impedance"), (width / 2, "position")):
        controller_rows = [row for row in rows if row["controller"] == controller_kind]
        success = [int(bool(row["success"])) for row in controller_rows]
        force = [float(row["max_contact_force_n"]) for row in controller_rows]
        axes[0].bar(
            indices + offset,
            success,
            width,
            color=colors[controller_kind],
            label=controller_kind,
        )
        axes[1].bar(
            indices + offset,
            force,
            width,
            color=colors[controller_kind],
            label=controller_kind,
        )
    axes[0].set_ylabel("Insertion success")
    axes[0].set_yticks([0, 1], labels=["fail", "pass"])
    axes[0].set_ylim(0.0, 1.15)
    axes[1].set_ylabel("Peak contact force [N]")
    axes[1].set_xticks(indices, labels=scenarios, rotation=25, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def run_controller_robustness_comparison(config: ProjectConfig, output: Path | None = None) -> Path:
    """Compare both Cartesian controllers across the fixed disturbance matrix."""
    directory = _output_directory(config, "controller_robustness", output)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scenario, variation in ROBUSTNESS_SCENARIOS:
        for controller_kind in ("impedance", "position"):
            summary, _ = run_peg(
                config,
                viewer=False,
                output=directory / scenario / controller_kind,
                variation=variation,
                controller_kind=controller_kind,
            )
            rows.append(
                {
                    "scenario": scenario,
                    **asdict(variation),
                    **asdict(summary),
                }
            )

    with (directory / "robustness_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    controller_rows = {
        controller_kind: [row for row in rows if row["controller"] == controller_kind]
        for controller_kind in ("impedance", "position")
    }
    aggregates = {
        controller_kind: _aggregate_controller_rows(values)
        for controller_kind, values in controller_rows.items()
    }
    comparison = {
        "model_variant": config.model.variant,
        "scenarios": len(ROBUSTNESS_SCENARIOS),
        "aggregates": aggregates,
        "position_minus_impedance_success_rate": (
            float(aggregates["position"]["success_rate"])
            - float(aggregates["impedance"]["success_rate"])
        ),
        "position_over_impedance_mean_peak_force_ratio": (
            float(aggregates["position"]["mean_peak_contact_force_n"])
            / float(aggregates["impedance"]["mean_peak_contact_force_n"])
        ),
        "trials": rows,
    }
    with (directory / "robustness_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    if config.output.save_plot:
        _save_robustness_comparison_plot(rows, directory / "robustness_comparison.png")
    return directory
