from __future__ import annotations

import csv
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
    JointPDController,
    apply_torque,
    site_pose,
)
from panda_assembly.math_utils import smoothstep
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
    model = load_model(timestep=config.simulation.timestep)
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
    )


def run_cartesian(
    config: ProjectConfig,
    viewer: bool = False,
    output: Path | None = None,
) -> tuple[RunSummary, Path]:
    model = load_model(timestep=config.simulation.timestep)
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    start_position, desired_rotation = site_pose(model, data, ee_id)
    goal = start_position + np.asarray(config.cartesian_controller.target_offset)
    controller = _cartesian_controller(config, data.qpos[:7])
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
        success=final_error < 0.015,
        completion_time=config.simulation.duration,
        output=output,
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
) -> tuple[RunSummary, Path]:
    variation = variation or Variation()
    model = load_model(timestep=config.simulation.timestep)
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    ee_id, peg_id, hole_id = _configure_variation(
        model, data, variation, config.peg_task.nominal_friction
    )
    start_position, desired_rotation = site_pose(model, data, ee_id)
    hole_position, _ = site_pose(model, data, hole_id)
    commanded_hole_position = hole_position.copy()
    commanded_hole_position[0] -= variation.lateral_offset_x
    commanded_hole_position[1] -= variation.lateral_offset_y
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
    initial_tip, _ = site_pose(model, data, tip_id)
    controller = _cartesian_controller(config, data.qpos[:7])
    peg = config.peg_task
    log = RunLog()
    completion_time: float | None = None

    aligned = np.array([commanded_hole_position[0], commanded_hole_position[1], start_position[2]])
    final_target = aligned.copy()
    final_target[2] -= max(0.0, initial_tip[2] - hole_position[2]) + peg.insertion_depth

    def desired_position(sim_time: float) -> np.ndarray:
        if sim_time <= peg.approach_time:
            return start_position
        if sim_time <= peg.align_time:
            progress = (sim_time - peg.approach_time) / (peg.align_time - peg.approach_time)
            return start_position + smoothstep(progress) * (aligned - start_position)
        progress = (sim_time - peg.align_time) / (peg.insert_time - peg.align_time)
        return aligned + smoothstep(progress) * (final_target - aligned)

    def step(sim_time: float) -> None:
        nonlocal completion_time
        target = desired_position(sim_time)
        torque, error = controller.torque(model, data, ee_id, target, desired_rotation)
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
    )


def run_robustness_batch(
    config: ProjectConfig,
    output: Path | None = None,
) -> Path:
    directory = _output_directory(config, "batch", output)
    directory.mkdir(parents=True, exist_ok=True)
    variations = [
        Variation(),
        Variation(lateral_offset_x=0.002),
        Variation(lateral_offset_y=-0.002),
        Variation(friction_scale=0.6),
        Variation(friction_scale=1.4),
        Variation(tool_mass_scale=0.8),
        Variation(tool_mass_scale=1.2),
        Variation(lateral_offset_x=0.002, friction_scale=1.4, tool_mass_scale=1.2),
    ]
    rows: list[dict[str, object]] = []
    for index, variation in enumerate(variations):
        summary, _ = run_peg(
            config,
            viewer=False,
            output=directory / f"trial_{index:02d}",
            variation=variation,
        )
        rows.append({**asdict(variation), **asdict(summary)})

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
    import json

    with (directory / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    return directory
