from __future__ import annotations

import csv
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
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
    minimum_geom_proximity,
    save_plot,
    save_summary,
    total_contact_force,
)
from panda_assembly.model import load_model, reset_to_home
from panda_assembly.safety import (
    ForceSafetySupervisor,
    InsertionSpeedLimiter,
    LateralCommandLimiter,
    PredictiveImpactGuard,
    SpiralRecoverySupervisor,
    classify_peg_outcome,
)
from panda_assembly.trajectories import quintic_joint_trajectory


@dataclass(frozen=True)
class Variation:
    lateral_offset_x: float = 0.0
    lateral_offset_y: float = 0.0
    friction_scale: float = 1.0
    tool_mass_scale: float = 1.0


@dataclass(frozen=True)
class BoundaryScanPoint:
    scenario: str
    direction: str
    direction_label: str
    angle_deg: float | None
    offset_m: float
    variation: Variation


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


_SQRT_HALF = 0.7071067811865476
BOUNDARY_DIRECTIONS = (
    ("x_pos", "+X", 0.0, 1.0, 0.0),
    ("xy_pos", "+X+Y", 45.0, _SQRT_HALF, _SQRT_HALF),
    ("y_pos", "+Y", 90.0, 0.0, 1.0),
    ("x_neg_y_pos", "-X+Y", 135.0, -_SQRT_HALF, _SQRT_HALF),
    ("x_neg", "-X", 180.0, -1.0, 0.0),
    ("xy_neg", "-X-Y", 225.0, -_SQRT_HALF, -_SQRT_HALF),
    ("y_neg", "-Y", 270.0, 0.0, -1.0),
    ("x_pos_y_neg", "+X-Y", 315.0, _SQRT_HALF, -_SQRT_HALF),
)


SAFETY_RECOVERY_SCENARIOS = (
    ("nominal", "nominal", Variation()),
    ("x_plus_6mm", "+X 6 mm", Variation(lateral_offset_x=0.006)),
    (
        "xy_plus_8mm",
        "+X+Y 8 mm",
        Variation(
            lateral_offset_x=0.008 * _SQRT_HALF,
            lateral_offset_y=0.008 * _SQRT_HALF,
        ),
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
    outcome: str | None = None,
    failure_type: str | None = None,
    safety_triggered: bool = False,
    safety_trigger_time: float | None = None,
    safety_trigger_force: float | None = None,
    retraction_distance: float | None = None,
    safety_recovered: bool = False,
    contact_recovery_triggered: bool = False,
    jam_detected_time: float | None = None,
    recovery_attempts: int = 0,
    spiral_entry_detected: bool = False,
    retry_success: bool = False,
    recovery_safe_retreat: bool = False,
    predictive_guard_triggered: bool = False,
    predictive_trigger_time: float | None = None,
    predictive_trigger_reason: str | None = None,
    predictive_trigger_distance: float | None = None,
    predictive_trigger_axis_component: float | None = None,
    predictive_trigger_lateral_gap: float | None = None,
    minimum_proximity_distance: float | None = None,
    max_force_rate: float | None = None,
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
        outcome=outcome,
        failure_type=failure_type,
        final_contact_force_n=log.final_contact_force(),
        safety_triggered=safety_triggered,
        safety_trigger_time_s=safety_trigger_time,
        safety_trigger_force_n=safety_trigger_force,
        retraction_distance_m=retraction_distance,
        safety_recovered=safety_recovered,
        contact_recovery_triggered=contact_recovery_triggered,
        jam_detected_time_s=jam_detected_time,
        recovery_attempts=recovery_attempts,
        spiral_entry_detected=spiral_entry_detected,
        retry_success=retry_success,
        recovery_safe_retreat=recovery_safe_retreat,
        predictive_guard_triggered=predictive_guard_triggered,
        predictive_trigger_time_s=predictive_trigger_time,
        predictive_trigger_reason=predictive_trigger_reason,
        predictive_trigger_distance_m=predictive_trigger_distance,
        predictive_trigger_axis_component=predictive_trigger_axis_component,
        predictive_trigger_lateral_gap_m=predictive_trigger_lateral_gap,
        minimum_proximity_distance_m=minimum_proximity_distance,
        max_force_rate_n_s=max_force_rate,
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
    force_safety: bool = False,
    contact_recovery: bool = False,
    predictive_impact: bool = False,
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
    socket_wall_geom_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "socket_x_pos",
            "socket_x_neg",
            "socket_y_pos",
            "socket_y_neg",
        )
    )
    vertical_rotation = vertical_tool_frame(start_rotation)
    tip_offset_local = _site_offset_in_frame(model, data, ee_id, tip_id)
    vertical_tip_offset = vertical_rotation @ tip_offset_local
    controller = _task_controller(config, data.qpos[:7], controller_kind)
    peg = config.peg_task
    log = RunLog()
    completion_time: float | None = None
    safety = config.force_safety
    supervisor = (
        ForceSafetySupervisor(
            force_limit_n=safety.force_limit_n,
            trigger_duration_s=safety.trigger_duration_s,
            hold_time_s=safety.hold_time_s,
            retract_distance_m=safety.retract_distance_m,
            retract_duration_s=safety.retract_duration_s,
        )
        if force_safety
        else None
    )

    aligned = np.array(
        [
            commanded_hole_position[0] - vertical_tip_offset[0],
            commanded_hole_position[1] - vertical_tip_offset[1],
            start_position[2],
        ]
    )
    final_target = aligned.copy()
    final_target[2] = commanded_hole_position[2] - peg.target_depth - vertical_tip_offset[2]
    recovery_values = config.contact_recovery
    speed_limiter = (
        InsertionSpeedLimiter(
            timestep_s=config.simulation.timestep,
            precontact_distance_m=recovery_values.precontact_distance_m,
            slow_zone_depth_m=recovery_values.slow_zone_depth_m,
            free_speed_m_s=recovery_values.free_speed_m_s,
            precontact_speed_m_s=recovery_values.precontact_speed_m_s,
            contact_slow_force_n=recovery_values.contact_slow_force_n,
            contact_speed_m_s=recovery_values.contact_speed_m_s,
        )
        if contact_recovery
        else None
    )
    recovery_supervisor = (
        SpiralRecoverySupervisor(
            jam_force_n=recovery_values.jam_force_n,
            jam_timeout_s=recovery_values.jam_timeout_s,
            jam_progress_m=recovery_values.jam_progress_m,
            retract_distance_m=recovery_values.retract_distance_m,
            retract_duration_s=recovery_values.retract_duration_s,
            spiral_radius_m=recovery_values.spiral_radius_m,
            spiral_cycles=recovery_values.spiral_cycles,
            spiral_duration_s=recovery_values.spiral_duration_s,
            spiral_descent_m=recovery_values.spiral_descent_m,
            entry_depth_m=recovery_values.entry_depth_m,
            retry_insert_duration_s=recovery_values.retry_insert_duration_s,
            max_retries=recovery_values.max_retries,
            search_center=aligned.copy(),
            search_target_z_m=float(
                commanded_hole_position[2]
                - 2.0 * recovery_values.entry_depth_m
                - vertical_tip_offset[2]
            ),
            final_target_z_m=float(final_target[2]),
        )
        if contact_recovery
        else None
    )
    predictive_values = config.predictive_impact
    predictive_guard = (
        PredictiveImpactGuard(
            timestep_s=config.simulation.timestep,
            slowdown_distance_m=predictive_values.slowdown_distance_m,
            trigger_distance_m=predictive_values.trigger_distance_m,
            minimum_closing_speed_m_s=predictive_values.minimum_closing_speed_m_s,
            minimum_approach_axis_component=(predictive_values.minimum_approach_axis_component),
            maximum_lateral_gap_m=predictive_values.maximum_lateral_gap_m,
            probe_speed_m_s=predictive_values.probe_speed_m_s,
            force_rate_limit_n_s=predictive_values.force_rate_limit_n_s,
            force_rate_min_force_n=predictive_values.force_rate_min_force_n,
        )
        if predictive_impact
        else None
    )
    lateral_limiter = (
        LateralCommandLimiter(
            timestep_s=config.simulation.timestep,
            speed_limit_m_s=predictive_values.lateral_speed_limit_m_s,
        )
        if predictive_impact
        else None
    )

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
        force = total_contact_force(model, data, peg_id)
        current_position, current_rotation = site_pose(model, data, ee_id)
        tip_position, _ = site_pose(model, data, tip_id)
        xy_error = float(np.linalg.norm(tip_position[:2] - hole_position[:2]))
        depth = float(hole_position[2] - tip_position[2])
        if completion_time is None and xy_error <= peg.clearance and depth >= peg.insertion_depth:
            completion_time = sim_time

        target = desired_position(sim_time)
        target_rotation = desired_rotation(sim_time)
        insertion_active = sim_time > peg.align_time and completion_time is None
        proximity_distance = None
        proximity_axis_component = 0.0
        proximity_lateral_gap = predictive_values.distance_cutoff_m
        predictive_triggered_now = False
        predictive_state = "disabled"
        if predictive_guard is not None:
            (
                proximity_distance,
                proximity_axis_component,
                proximity_lateral_gap,
            ) = minimum_geom_proximity(
                model,
                data,
                peg_id,
                socket_wall_geom_ids,
                predictive_values.distance_cutoff_m,
            )
            prediction_active = bool(
                insertion_active
                and recovery_supervisor is not None
                and recovery_supervisor.state_name == "monitoring"
                and not (supervisor is not None and supervisor.triggered)
            )
            predictive_triggered_now = predictive_guard.update(
                sim_time,
                proximity_distance,
                proximity_axis_component,
                proximity_lateral_gap,
                force,
                prediction_active,
            )
            predictive_state = predictive_guard.state_name
        speed_state = "disabled"
        recovery_state = "disabled"
        if speed_limiter is not None:
            target, speed_state = speed_limiter.command(
                target,
                tip_z_m=float(tip_position[2]),
                hole_z_m=float(hole_position[2]),
                contact_force_n=force,
                insertion_active=insertion_active,
                speed_cap_m_s=(
                    predictive_guard.speed_cap_m_s if predictive_guard is not None else None
                ),
            )

        safety_state = "disabled"
        if supervisor is not None:
            just_triggered = supervisor.update(
                sim_time,
                force,
                current_position,
                current_rotation,
            )
            if just_triggered and isinstance(controller, CartesianPositionController):
                controller.desired_joint_position[:] = data.qpos[:7]

        if recovery_supervisor is not None and not (
            supervisor is not None and supervisor.triggered
        ):
            predictive_recovery = False
            if predictive_triggered_now:
                predictive_recovery = recovery_supervisor.trigger_recovery(
                    sim_time,
                    current_position,
                    current_rotation,
                )
            jam_triggered = recovery_supervisor.update(
                sim_time,
                force,
                depth,
                completion_time is not None,
                current_position,
                current_rotation,
            )
            if (predictive_recovery or jam_triggered) and isinstance(
                controller, CartesianPositionController
            ):
                controller.desired_joint_position[:] = data.qpos[:7]
            target, target_rotation, recovery_state = recovery_supervisor.command(
                sim_time,
                target,
                target_rotation,
            )

        lateral_state = "disabled"
        if lateral_limiter is not None:
            target, lateral_state = lateral_limiter.command(
                target,
                recovery_state in {"spiral_search", "retry_insert"},
            )

        if supervisor is not None:
            target, target_rotation, safety_state = supervisor.command(
                sim_time,
                target,
                target_rotation,
            )
        torque, error = controller.torque(model, data, ee_id, target, target_rotation)
        apply_torque(model, data, torque)
        log.append(
            sim_time,
            error[:3],
            force,
            data.qpos,
            orientation_error=error[3:],
            safety_state=(
                f"force:{safety_state}|speed:{speed_state}|predictive:{predictive_state}"
                f"|recovery:{recovery_state}|lateral:{lateral_state}"
            ),
            proximity_distance=proximity_distance,
            proximity_axis_component=(
                predictive_guard.last_approach_axis_component
                if predictive_guard is not None
                else None
            ),
            proximity_lateral_gap=(
                predictive_guard.last_lateral_gap_m if predictive_guard is not None else None
            ),
            closing_speed=(
                predictive_guard.last_closing_speed_m_s if predictive_guard is not None else None
            ),
            force_rate=(
                predictive_guard.last_force_rate_n_s if predictive_guard is not None else None
            ),
        )

    duration = max(config.simulation.duration, peg.settle_time)
    if contact_recovery:
        duration = max(
            duration,
            peg.settle_time
            + recovery_values.jam_timeout_s
            + recovery_values.retract_duration_s
            + recovery_values.spiral_duration_s
            + recovery_values.retry_insert_duration_s
            + 2.0,
        )
    if predictive_impact:
        duration += 4.0
    _run_loop(model, data, duration, config.simulation.render_hz, step, viewer)
    final_tip, _ = site_pose(model, data, tip_id)
    final_xy_error = float(np.linalg.norm(final_tip[:2] - hole_position[:2]))
    final_depth = float(hole_position[2] - final_tip[2])
    final_ee, _ = site_pose(model, data, ee_id)
    safety_triggered = supervisor is not None and supervisor.triggered
    retraction_distance = None
    if safety_triggered:
        assert supervisor is not None
        assert supervisor.trigger_position is not None
        retraction_distance = max(0.0, float(final_ee[2] - supervisor.trigger_position[2]))
    safety_recovered = bool(
        safety_triggered
        and log.final_contact_force() <= safety.release_force_n
        and retraction_distance is not None
        and retraction_distance >= safety.minimum_retraction_m
    )
    contact_recovery_triggered = bool(
        recovery_supervisor is not None and recovery_supervisor.triggered
    )
    retry_success = bool(recovery_supervisor is not None and recovery_supervisor.retry_success)
    recovery_safe_retreat = bool(
        predictive_guard is not None
        and predictive_guard.triggered
        and recovery_supervisor is not None
        and recovery_supervisor.state_name == "recovery_failed"
        and final_depth <= 0.0
        and log.final_contact_force() <= safety.release_force_n
        and log.max_contact_force() <= safety.force_limit_n
    )
    geometric_success = completion_time is not None
    outcome, failure_type = classify_peg_outcome(
        geometric_success=geometric_success,
        max_contact_force_n=log.max_contact_force(),
        final_contact_force_n=log.final_contact_force(),
        final_xy_error_m=final_xy_error,
        final_insertion_depth_m=final_depth,
        clearance_m=peg.clearance,
        insertion_depth_m=peg.insertion_depth,
        force_limit_n=safety.force_limit_n,
        safety_triggered=safety_triggered,
        safety_recovered=safety_recovered,
        contact_recovery_triggered=contact_recovery_triggered,
        retry_success=retry_success,
        predictive_guard_triggered=bool(
            predictive_guard is not None and predictive_guard.triggered
        ),
        recovery_safe_retreat=recovery_safe_retreat,
    )
    return _finish_run(
        config,
        "peg",
        log,
        success=geometric_success and not safety_triggered,
        completion_time=completion_time,
        output=output,
        final_xy_error=final_xy_error,
        final_depth=final_depth,
        controller=controller_kind,
        outcome=outcome,
        failure_type=failure_type,
        safety_triggered=safety_triggered,
        safety_trigger_time=(supervisor.trigger_time_s if supervisor is not None else None),
        safety_trigger_force=(supervisor.trigger_force_n if supervisor is not None else None),
        retraction_distance=retraction_distance,
        safety_recovered=safety_recovered,
        contact_recovery_triggered=contact_recovery_triggered,
        jam_detected_time=(
            recovery_supervisor.trigger_time_s if recovery_supervisor is not None else None
        ),
        recovery_attempts=(recovery_supervisor.attempts if recovery_supervisor is not None else 0),
        spiral_entry_detected=(
            recovery_supervisor.entry_detected if recovery_supervisor is not None else False
        ),
        retry_success=retry_success,
        recovery_safe_retreat=recovery_safe_retreat,
        predictive_guard_triggered=bool(
            predictive_guard is not None and predictive_guard.triggered
        ),
        predictive_trigger_time=(
            predictive_guard.trigger_time_s if predictive_guard is not None else None
        ),
        predictive_trigger_reason=(
            predictive_guard.trigger_reason if predictive_guard is not None else None
        ),
        predictive_trigger_distance=(
            predictive_guard.triggered_at_distance_m if predictive_guard is not None else None
        ),
        predictive_trigger_axis_component=(
            predictive_guard.triggered_at_axis_component if predictive_guard is not None else None
        ),
        predictive_trigger_lateral_gap=(
            predictive_guard.triggered_at_lateral_gap_m if predictive_guard is not None else None
        ),
        minimum_proximity_distance=(
            predictive_guard.minimum_distance_m if predictive_guard is not None else None
        ),
        max_force_rate=(
            predictive_guard.maximum_force_rate_n_s if predictive_guard is not None else None
        ),
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


def _monte_carlo_variations(config: ProjectConfig) -> tuple[Variation, ...]:
    values = config.monte_carlo
    generator = np.random.default_rng(values.seed)
    return tuple(
        Variation(
            lateral_offset_x=float(
                generator.uniform(-values.lateral_offset_limit_m, values.lateral_offset_limit_m)
            ),
            lateral_offset_y=float(
                generator.uniform(-values.lateral_offset_limit_m, values.lateral_offset_limit_m)
            ),
            friction_scale=float(
                generator.uniform(values.friction_scale_min, values.friction_scale_max)
            ),
            tool_mass_scale=float(
                generator.uniform(values.tool_mass_scale_min, values.tool_mass_scale_max)
            ),
        )
        for _ in range(values.trials)
    )


def wilson_score_interval(
    successes: int, trials: int, z_score: float = 1.96
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial success rate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    proportion = successes / trials
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z_score
        * np.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return float(center - margin), float(center + margin)


def _mean_confidence_interval(values: list[float]) -> tuple[float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean
    margin = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean - margin, mean + margin


def _monte_carlo_aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    successes = sum(bool(row["success"]) for row in rows)
    forces = [float(row["max_contact_force_n"]) for row in rows]
    completion_times = [
        float(row["completion_time_s"]) for row in rows if row["completion_time_s"] is not None
    ]
    return {
        "trials": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "success_rate_wilson_ci95": list(wilson_score_interval(successes, len(rows))),
        "mean_peak_contact_force_n": float(np.mean(forces)),
        "mean_peak_contact_force_ci95_n": list(_mean_confidence_interval(forces)),
        "max_peak_contact_force_n": float(np.max(forces)),
        "mean_completion_time_s_successes": (
            float(np.mean(completion_times)) if completion_times else None
        ),
    }


def _save_monte_carlo_plot(
    rows: list[dict[str, object]], aggregates: dict[str, dict[str, object]], path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    controller_names = ("impedance", "position")
    colors = {"impedance": "#2166ac", "position": "#b2182b"}
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))

    success_rates = [float(aggregates[name]["success_rate"]) for name in controller_names]
    intervals = [aggregates[name]["success_rate_wilson_ci95"] for name in controller_names]
    lower_errors = [
        rate - float(interval[0]) for rate, interval in zip(success_rates, intervals, strict=True)
    ]
    upper_errors = [
        float(interval[1]) - rate for rate, interval in zip(success_rates, intervals, strict=True)
    ]
    axes[0].bar(
        controller_names,
        success_rates,
        color=[colors[name] for name in controller_names],
        yerr=np.asarray([lower_errors, upper_errors]),
        capsize=6,
    )
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Success rate with Wilson 95% CI")
    axes[0].grid(axis="y", alpha=0.25)

    for controller_kind in controller_names:
        controller_rows = [row for row in rows if row["controller"] == controller_kind]
        controller_rows.sort(key=lambda row: float(row["lateral_offset_norm_m"]))
        offsets = [1000.0 * float(row["lateral_offset_norm_m"]) for row in controller_rows]
        forces = [float(row["max_contact_force_n"]) for row in controller_rows]
        axes[1].plot(
            offsets,
            forces,
            marker="o",
            color=colors[controller_kind],
            label=controller_kind,
            alpha=0.8,
        )
        failed_offsets = [
            1000.0 * float(row["lateral_offset_norm_m"])
            for row in controller_rows
            if not bool(row["success"])
        ]
        failed_forces = [
            float(row["max_contact_force_n"]) for row in controller_rows if not bool(row["success"])
        ]
        axes[1].scatter(
            failed_offsets,
            failed_forces,
            marker="x",
            s=70,
            linewidths=2,
            color=colors[controller_kind],
        )
    axes[1].set_xlabel("Lateral fixture error norm [mm]")
    axes[1].set_ylabel("Peak contact force [N]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def run_monte_carlo_comparison(config: ProjectConfig, output: Path | None = None) -> Path:
    """Run paired, reproducible random disturbances for both Cartesian controllers."""
    directory = _output_directory(config, "monte_carlo", output)
    directory.mkdir(parents=True, exist_ok=True)
    trial_config = replace(config, output=replace(config.output, save_plot=False))
    rows: list[dict[str, object]] = []
    for index, variation in enumerate(_monte_carlo_variations(config)):
        trial_name = f"trial_{index:03d}"
        offset_norm = float(np.hypot(variation.lateral_offset_x, variation.lateral_offset_y))
        for controller_kind in ("impedance", "position"):
            summary, _ = run_peg(
                trial_config,
                viewer=False,
                output=directory / trial_name / controller_kind,
                variation=variation,
                controller_kind=controller_kind,
            )
            rows.append(
                {
                    "trial": trial_name,
                    "seed": config.monte_carlo.seed,
                    "lateral_offset_norm_m": offset_norm,
                    **asdict(variation),
                    **asdict(summary),
                }
            )

    with (directory / "monte_carlo_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    controller_rows = {
        controller_kind: [row for row in rows if row["controller"] == controller_kind]
        for controller_kind in ("impedance", "position")
    }
    aggregates = {
        controller_kind: _monte_carlo_aggregate(values)
        for controller_kind, values in controller_rows.items()
    }
    paired_outcomes = {"both_succeed": 0, "impedance_only": 0, "position_only": 0, "neither": 0}
    paired_force_differences: list[float] = []
    for index in range(config.monte_carlo.trials):
        impedance = controller_rows["impedance"][index]
        position = controller_rows["position"][index]
        impedance_success = bool(impedance["success"])
        position_success = bool(position["success"])
        if impedance_success and position_success:
            paired_outcomes["both_succeed"] += 1
        elif impedance_success:
            paired_outcomes["impedance_only"] += 1
        elif position_success:
            paired_outcomes["position_only"] += 1
        else:
            paired_outcomes["neither"] += 1
        paired_force_differences.append(
            float(position["max_contact_force_n"]) - float(impedance["max_contact_force_n"])
        )

    result = {
        "model_variant": config.model.variant,
        "seed": config.monte_carlo.seed,
        "sampling": asdict(config.monte_carlo),
        "aggregates": aggregates,
        "paired_outcomes": paired_outcomes,
        "mean_paired_position_minus_impedance_peak_force_n": float(
            np.mean(paired_force_differences)
        ),
        "trials": rows,
    }
    with (directory / "monte_carlo_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if config.output.save_plot:
        _save_monte_carlo_plot(rows, aggregates, directory / "monte_carlo_summary.png")
    return directory


def _boundary_scan_points(config: ProjectConfig) -> tuple[BoundaryScanPoint, ...]:
    """Return the unique origin and eight-direction radial scan points."""
    scan = config.boundary_scan
    steps = int(round(scan.max_offset_m / scan.step_m))
    points = [
        BoundaryScanPoint(
            scenario="origin",
            direction="origin",
            direction_label="origin",
            angle_deg=None,
            offset_m=0.0,
            variation=Variation(),
        )
    ]
    for step_index in range(1, steps + 1):
        offset = step_index * scan.step_m
        offset_mm = int(round(1000.0 * offset))
        for direction, label, angle_deg, x_unit, y_unit in BOUNDARY_DIRECTIONS:
            points.append(
                BoundaryScanPoint(
                    scenario=f"{direction}_{offset_mm:03d}mm",
                    direction=direction,
                    direction_label=label,
                    angle_deg=angle_deg,
                    offset_m=offset,
                    variation=Variation(
                        lateral_offset_x=offset * x_unit,
                        lateral_offset_y=offset * y_unit,
                    ),
                )
            )
    return tuple(points)


def _direction_boundary_summary(
    controller_rows: list[dict[str, object]],
    direction: str,
    success_key: str = "success",
) -> dict[str, object]:
    origin = next(row for row in controller_rows if row["direction"] == "origin")
    rows = [origin] + [row for row in controller_rows if row["direction"] == direction]
    rows.sort(key=lambda row: float(row["offset_m"]))
    first_failure_index = next(
        (index for index, row in enumerate(rows) if not bool(row[success_key])), None
    )
    successful_offsets = [float(row["offset_m"]) for row in rows if bool(row[success_key])]
    if first_failure_index is None:
        contiguous_limit = float(rows[-1]["offset_m"])
        first_failure_offset = None
        non_monotonic = False
    else:
        contiguous_limit = (
            float(rows[first_failure_index - 1]["offset_m"]) if first_failure_index > 0 else None
        )
        first_failure_offset = float(rows[first_failure_index]["offset_m"])
        non_monotonic = any(bool(row[success_key]) for row in rows[first_failure_index + 1 :])
    return {
        "origin_success": bool(origin[success_key]),
        "contiguous_success_limit_m": contiguous_limit,
        "first_failure_offset_m": first_failure_offset,
        "boundary_bracket_m": (
            [contiguous_limit, first_failure_offset]
            if contiguous_limit is not None and first_failure_offset is not None
            else None
        ),
        "maximum_successful_offset_m": max(successful_offsets) if successful_offsets else None,
        "non_monotonic_success": non_monotonic,
        "max_peak_contact_force_n": max(float(row["max_contact_force_n"]) for row in rows),
    }


def _boundary_controller_summary(
    controller_rows: list[dict[str, object]], config: ProjectConfig
) -> dict[str, object]:
    directions = {
        direction: {
            "label": label,
            "angle_deg": angle_deg,
            **_direction_boundary_summary(controller_rows, direction),
        }
        for direction, label, angle_deg, _, _ in BOUNDARY_DIRECTIONS
    }
    steps = int(round(config.boundary_scan.max_offset_m / config.boundary_scan.step_m))
    offset_summaries: list[dict[str, object]] = []
    for step_index in range(steps + 1):
        offset = step_index * config.boundary_scan.step_m
        offset_rows = [
            row for row in controller_rows if abs(float(row["offset_m"]) - offset) < 1e-12
        ]
        successes = sum(bool(row["success"]) for row in offset_rows)
        offset_summaries.append(
            {
                "offset_m": offset,
                "evaluations": len(offset_rows),
                "successes": successes,
                "success_fraction": successes / len(offset_rows),
                "worst_peak_contact_force_n": max(
                    float(row["max_contact_force_n"]) for row in offset_rows
                ),
            }
        )
    limits = [float(values["contiguous_success_limit_m"] or 0.0) for values in directions.values()]
    return {
        "evaluations": len(controller_rows),
        "successes": sum(bool(row["success"]) for row in controller_rows),
        "origin_success": bool(
            next(row["success"] for row in controller_rows if row["direction"] == "origin")
        ),
        "isotropic_contiguous_success_radius_m": min(limits),
        "best_direction_contiguous_success_limit_m": max(limits),
        "non_monotonic_directions": [
            direction
            for direction, values in directions.items()
            if bool(values["non_monotonic_success"])
        ],
        "directions": directions,
        "offset_summary": offset_summaries,
    }


def _save_boundary_scan_plot(
    aggregates: dict[str, dict[str, object]], config: ProjectConfig, path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    colors = {"impedance": "#2166ac", "position": "#b2182b"}
    direction_angles = np.radians([values[2] for values in BOUNDARY_DIRECTIONS])
    closed_angles = np.append(direction_angles, direction_angles[0])
    figure = plt.figure(figsize=(16, 5.5))
    polar_axis = figure.add_subplot(1, 3, 1, projection="polar")
    success_axis = figure.add_subplot(1, 3, 2)
    force_axis = figure.add_subplot(1, 3, 3)

    for controller_kind in ("impedance", "position"):
        aggregate = aggregates[controller_kind]
        direction_values = aggregate["directions"]
        limits = [
            1000.0 * float(direction_values[direction]["contiguous_success_limit_m"] or 0.0)
            for direction, _, _, _, _ in BOUNDARY_DIRECTIONS
        ]
        closed_limits = np.append(limits, limits[0])
        polar_axis.plot(
            closed_angles,
            closed_limits,
            marker="o",
            color=colors[controller_kind],
            label=controller_kind,
        )
        polar_axis.fill(closed_angles, closed_limits, color=colors[controller_kind], alpha=0.08)

        offset_summary = aggregate["offset_summary"]
        offsets_mm = [1000.0 * float(row["offset_m"]) for row in offset_summary]
        success_fractions = [float(row["success_fraction"]) for row in offset_summary]
        worst_forces = [float(row["worst_peak_contact_force_n"]) for row in offset_summary]
        success_axis.plot(
            offsets_mm,
            success_fractions,
            marker="o",
            color=colors[controller_kind],
            label=controller_kind,
        )
        force_axis.plot(
            offsets_mm,
            worst_forces,
            marker="o",
            color=colors[controller_kind],
            label=controller_kind,
        )

    polar_axis.set_theta_zero_location("E")
    polar_axis.set_theta_direction(1)
    polar_axis.set_xticks(
        direction_angles,
        labels=[values[1] for values in BOUNDARY_DIRECTIONS],
    )
    polar_axis.set_ylim(0.0, 1000.0 * config.boundary_scan.max_offset_m)
    polar_axis.set_title("Contiguous success boundary [mm]")
    polar_axis.grid(alpha=0.3)
    polar_axis.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))

    success_axis.set_xlabel("Fixture offset [mm]")
    success_axis.set_ylabel("Successful directions / tested directions")
    success_axis.set_ylim(-0.04, 1.04)
    success_axis.grid(alpha=0.25)
    success_axis.legend()

    force_axis.set_xlabel("Fixture offset [mm]")
    force_axis.set_ylabel("Worst peak contact force [N]")
    force_axis.set_yscale("log")
    force_axis.grid(alpha=0.25, which="both")
    force_axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def run_controller_boundary_scan(config: ProjectConfig, output: Path | None = None) -> Path:
    """Map directional insertion limits for both controllers at nominal friction and mass."""
    directory = _output_directory(config, "boundary_scan", output)
    directory.mkdir(parents=True, exist_ok=True)
    trial_config = replace(
        config,
        output=replace(config.output, save_csv=False, save_plot=False),
    )
    rows: list[dict[str, object]] = []
    for point in _boundary_scan_points(config):
        for controller_kind in ("impedance", "position"):
            summary, _ = run_peg(
                trial_config,
                viewer=False,
                output=directory / point.scenario / controller_kind,
                variation=point.variation,
                controller_kind=controller_kind,
            )
            rows.append(
                {
                    "scenario": point.scenario,
                    "direction": point.direction,
                    "direction_label": point.direction_label,
                    "angle_deg": point.angle_deg,
                    "offset_m": point.offset_m,
                    **asdict(point.variation),
                    **asdict(summary),
                }
            )

    with (directory / "boundary_scan_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    controller_rows = {
        controller_kind: [row for row in rows if row["controller"] == controller_kind]
        for controller_kind in ("impedance", "position")
    }
    aggregates = {
        controller_kind: _boundary_controller_summary(values, config)
        for controller_kind, values in controller_rows.items()
    }
    result = {
        "model_variant": config.model.variant,
        "scan": asdict(config.boundary_scan),
        "directions": [
            {"name": name, "label": label, "angle_deg": angle_deg}
            for name, label, angle_deg, _, _ in BOUNDARY_DIRECTIONS
        ],
        "boundary_definition": (
            "largest tested radius for which every smaller tested radius in the same "
            "direction also succeeded"
        ),
        "nominal_friction_scale": 1.0,
        "nominal_tool_mass_scale": 1.0,
        "aggregates": aggregates,
        "trials": rows,
    }
    with (directory / "boundary_scan_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if config.output.save_plot:
        _save_boundary_scan_plot(aggregates, config, directory / "boundary_scan_summary.png")
    return directory


def _boundary_refinement_points(config: ProjectConfig) -> tuple[BoundaryScanPoint, ...]:
    """Return coarse anchors plus the fine transition-region radial grid."""
    refinement = config.boundary_refinement
    fine_steps = int(round((refinement.max_offset_m - refinement.min_offset_m) / refinement.step_m))
    fine_offsets = [
        refinement.min_offset_m + index * refinement.step_m for index in range(fine_steps + 1)
    ]
    anchor_offsets = [
        offset
        for offset in (config.boundary_scan.step_m,)
        if 0.0 < offset < refinement.min_offset_m
    ]
    offsets = sorted({round(value, 12) for value in anchor_offsets + fine_offsets})
    points = [
        BoundaryScanPoint(
            scenario="origin",
            direction="origin",
            direction_label="origin",
            angle_deg=None,
            offset_m=0.0,
            variation=Variation(),
        )
    ]
    for offset in offsets:
        offset_um = int(round(1_000_000.0 * offset))
        for direction, label, angle_deg, x_unit, y_unit in BOUNDARY_DIRECTIONS:
            points.append(
                BoundaryScanPoint(
                    scenario=f"{direction}_{offset_um:05d}um",
                    direction=direction,
                    direction_label=label,
                    angle_deg=angle_deg,
                    offset_m=offset,
                    variation=Variation(
                        lateral_offset_x=offset * x_unit,
                        lateral_offset_y=offset * y_unit,
                    ),
                )
            )
    return tuple(points)


def _boundary_refinement_controller_summary(
    controller_rows: list[dict[str, object]], config: ProjectConfig
) -> dict[str, object]:
    directions = {
        direction: {
            "label": label,
            "angle_deg": angle_deg,
            "geometric": _direction_boundary_summary(
                controller_rows,
                direction,
                success_key="success",
            ),
            "safe": _direction_boundary_summary(
                controller_rows,
                direction,
                success_key="safe_success",
            ),
        }
        for direction, label, angle_deg, _, _ in BOUNDARY_DIRECTIONS
    }
    offsets = sorted({float(row["offset_m"]) for row in controller_rows})
    offset_summaries: list[dict[str, object]] = []
    for offset in offsets:
        offset_rows = [
            row for row in controller_rows if abs(float(row["offset_m"]) - offset) < 1e-12
        ]
        geometric_successes = sum(bool(row["success"]) for row in offset_rows)
        safe_successes = sum(bool(row["safe_success"]) for row in offset_rows)
        offset_summaries.append(
            {
                "offset_m": offset,
                "evaluations": len(offset_rows),
                "geometric_successes": geometric_successes,
                "geometric_success_fraction": geometric_successes / len(offset_rows),
                "safe_successes": safe_successes,
                "safe_success_fraction": safe_successes / len(offset_rows),
                "worst_peak_contact_force_n": max(
                    float(row["max_contact_force_n"]) for row in offset_rows
                ),
            }
        )
    geometric_limits = [
        float(values["geometric"]["contiguous_success_limit_m"] or 0.0)
        for values in directions.values()
    ]
    safe_limits = [
        float(values["safe"]["contiguous_success_limit_m"] or 0.0) for values in directions.values()
    ]
    return {
        "evaluations": len(controller_rows),
        "geometric_successes": sum(bool(row["success"]) for row in controller_rows),
        "safe_successes": sum(bool(row["safe_success"]) for row in controller_rows),
        "safe_force_limit_n": config.boundary_refinement.safe_force_limit_n,
        "isotropic_geometric_success_radius_m": min(geometric_limits),
        "isotropic_safe_success_radius_m": min(safe_limits),
        "best_direction_geometric_success_limit_m": max(geometric_limits),
        "best_direction_safe_success_limit_m": max(safe_limits),
        "geometric_non_monotonic_directions": [
            direction
            for direction, values in directions.items()
            if bool(values["geometric"]["non_monotonic_success"])
        ],
        "safe_non_monotonic_directions": [
            direction
            for direction, values in directions.items()
            if bool(values["safe"]["non_monotonic_success"])
        ],
        "directions": directions,
        "offset_summary": offset_summaries,
    }


def _save_boundary_refinement_plot(
    aggregates: dict[str, dict[str, object]], config: ProjectConfig, path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    colors = {"impedance": "#2166ac", "position": "#b2182b"}
    styles = {"geometric": "-", "safe": "--"}
    direction_angles = np.radians([values[2] for values in BOUNDARY_DIRECTIONS])
    closed_angles = np.append(direction_angles, direction_angles[0])
    figure = plt.figure(figsize=(16, 5.5))
    polar_axis = figure.add_subplot(1, 3, 1, projection="polar")
    success_axis = figure.add_subplot(1, 3, 2)
    force_axis = figure.add_subplot(1, 3, 3)

    for controller_kind in ("impedance", "position"):
        aggregate = aggregates[controller_kind]
        direction_values = aggregate["directions"]
        for metric in ("geometric", "safe"):
            limits = [
                1000.0
                * float(direction_values[direction][metric]["contiguous_success_limit_m"] or 0.0)
                for direction, _, _, _, _ in BOUNDARY_DIRECTIONS
            ]
            closed_limits = np.append(limits, limits[0])
            polar_axis.plot(
                closed_angles,
                closed_limits,
                marker="o",
                color=colors[controller_kind],
                linestyle=styles[metric],
                label=f"{controller_kind} {metric}",
            )

        offset_summary = aggregate["offset_summary"]
        offsets_mm = [1000.0 * float(row["offset_m"]) for row in offset_summary]
        geometric_fractions = [float(row["geometric_success_fraction"]) for row in offset_summary]
        safe_fractions = [float(row["safe_success_fraction"]) for row in offset_summary]
        worst_forces = [float(row["worst_peak_contact_force_n"]) for row in offset_summary]
        success_axis.plot(
            offsets_mm,
            geometric_fractions,
            marker="o",
            color=colors[controller_kind],
            label=f"{controller_kind} geometric",
        )
        success_axis.plot(
            offsets_mm,
            safe_fractions,
            marker="x",
            color=colors[controller_kind],
            linestyle="--",
            label=f"{controller_kind} safe",
        )
        force_axis.plot(
            offsets_mm,
            worst_forces,
            marker="o",
            color=colors[controller_kind],
            label=controller_kind,
        )

    polar_axis.set_theta_zero_location("E")
    polar_axis.set_theta_direction(1)
    polar_axis.set_xticks(
        direction_angles,
        labels=[values[1] for values in BOUNDARY_DIRECTIONS],
    )
    polar_axis.set_ylim(0.0, 1000.0 * config.boundary_refinement.max_offset_m)
    polar_axis.set_title("Refined success boundary [mm]")
    polar_axis.grid(alpha=0.3)
    polar_axis.legend(loc="upper right", bbox_to_anchor=(1.35, 1.18), fontsize=8)

    success_axis.set_xlabel("Fixture offset [mm]")
    success_axis.set_ylabel("Successful directions / tested directions")
    success_axis.set_ylim(-0.04, 1.04)
    success_axis.grid(alpha=0.25)
    success_axis.legend(fontsize=8)

    force_axis.axhline(
        config.boundary_refinement.safe_force_limit_n,
        color="#333333",
        linestyle="--",
        label=f"safe-force limit ({config.boundary_refinement.safe_force_limit_n:g} N)",
    )
    force_axis.set_xlabel("Fixture offset [mm]")
    force_axis.set_ylabel("Worst peak contact force [N]")
    force_axis.set_yscale("log")
    force_axis.grid(alpha=0.25, which="both")
    force_axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def run_controller_boundary_refinement(config: ProjectConfig, output: Path | None = None) -> Path:
    """Refine the transition boundary and separate geometric from safe success."""
    directory = _output_directory(config, "boundary_refinement", output)
    directory.mkdir(parents=True, exist_ok=True)
    trial_config = replace(
        config,
        output=replace(config.output, save_csv=False, save_plot=False),
    )
    force_limit = config.boundary_refinement.safe_force_limit_n
    rows: list[dict[str, object]] = []
    for point in _boundary_refinement_points(config):
        for controller_kind in ("impedance", "position"):
            summary, _ = run_peg(
                trial_config,
                viewer=False,
                output=directory / point.scenario / controller_kind,
                variation=point.variation,
                controller_kind=controller_kind,
            )
            rows.append(
                {
                    "scenario": point.scenario,
                    "direction": point.direction,
                    "direction_label": point.direction_label,
                    "angle_deg": point.angle_deg,
                    "offset_m": point.offset_m,
                    **asdict(point.variation),
                    **asdict(summary),
                    "safe_force_limit_n": force_limit,
                    "safe_success": summary.success and summary.max_contact_force_n <= force_limit,
                }
            )

    with (directory / "boundary_refinement_trials.csv").open(
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
        controller_kind: _boundary_refinement_controller_summary(values, config)
        for controller_kind, values in controller_rows.items()
    }
    result = {
        "model_variant": config.model.variant,
        "refinement": asdict(config.boundary_refinement),
        "coarse_anchor_offset_m": config.boundary_scan.step_m,
        "directions": [
            {"name": name, "label": label, "angle_deg": angle_deg}
            for name, label, angle_deg, _, _ in BOUNDARY_DIRECTIONS
        ],
        "geometric_success_definition": (
            "configured insertion depth and lateral-clearance criteria are met"
        ),
        "safe_success_definition": (
            f"geometric success with peak simulated contact force no greater than {force_limit:g} N"
        ),
        "safe_force_limit_is_exploratory": True,
        "aggregates": aggregates,
        "trials": rows,
    }
    with (directory / "boundary_refinement_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if config.output.save_plot:
        _save_boundary_refinement_plot(
            aggregates,
            config,
            directory / "boundary_refinement_summary.png",
        )
    return directory


def _save_safety_recovery_plot(
    rows: list[dict[str, object]], config: ProjectConfig, path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [
        f"{scenario_label}\n{controller}"
        for _, scenario_label, _ in SAFETY_RECOVERY_SCENARIOS
        for controller in ("impedance", "position")
    ]
    pairs = [
        {
            policy: next(
                row
                for row in rows
                if row["scenario"] == scenario
                and row["controller"] == controller
                and row["policy"] == policy
            )
            for policy in ("unguarded", "guarded")
        }
        for scenario, _, _ in SAFETY_RECOVERY_SCENARIOS
        for controller in ("impedance", "position")
    ]
    indices = np.arange(len(pairs))
    width = 0.36
    colors = {"unguarded": "#b2182b", "guarded": "#2166ac"}
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))

    for offset, policy in ((-width / 2, "unguarded"), (width / 2, "guarded")):
        peak_forces = [max(0.1, float(pair[policy]["max_contact_force_n"])) for pair in pairs]
        final_forces = [max(0.1, float(pair[policy]["final_contact_force_n"])) for pair in pairs]
        axes[0, 0].bar(
            indices + offset,
            peak_forces,
            width,
            color=colors[policy],
            label=policy,
        )
        axes[0, 1].bar(
            indices + offset,
            final_forces,
            width,
            color=colors[policy],
            label=policy,
        )

    guarded_retractions = [
        1000.0 * float(pair["guarded"]["retraction_distance_m"] or 0.0) for pair in pairs
    ]
    retraction_colors = [
        "#1b7837" if bool(pair["guarded"]["safety_recovered"]) else "#bdbdbd" for pair in pairs
    ]
    axes[1, 0].bar(indices, guarded_retractions, color=retraction_colors)

    policies = ("unguarded", "guarded")
    outcome_counts = {
        policy: Counter(str(row["outcome"]) for row in rows if row["policy"] == policy)
        for policy in policies
    }
    outcomes = sorted({outcome for values in outcome_counts.values() for outcome in values})
    outcome_indices = np.arange(len(outcomes))
    for offset, policy in ((-width / 2, "unguarded"), (width / 2, "guarded")):
        axes[1, 1].bar(
            outcome_indices + offset,
            [outcome_counts[policy].get(outcome, 0) for outcome in outcomes],
            width,
            color=colors[policy],
            label=policy,
        )

    axes[0, 0].axhline(
        config.force_safety.force_limit_n,
        color="#333333",
        linestyle="--",
        label=f"trigger ({config.force_safety.force_limit_n:g} N)",
    )
    axes[0, 0].set_ylabel("Peak contact force [N]")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Peak load")
    axes[0, 1].axhline(
        config.force_safety.release_force_n,
        color="#333333",
        linestyle="--",
        label=f"released ({config.force_safety.release_force_n:g} N)",
    )
    axes[0, 1].set_ylabel("Final contact force [N]")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Load after recovery")
    axes[1, 0].axhline(
        1000.0 * config.force_safety.minimum_retraction_m,
        color="#333333",
        linestyle="--",
        label="minimum recovery distance",
    )
    axes[1, 0].set_ylabel("Guarded vertical retraction [mm]")
    axes[1, 0].set_title("Executed recovery")
    axes[1, 1].set_ylabel("Trials")
    axes[1, 1].set_title("Outcome taxonomy")
    axes[1, 1].set_xticks(outcome_indices, labels=outcomes, rotation=25, ha="right")
    axes[1, 1].set_yticks(range(0, len(rows) // 2 + 1))

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(indices, labels=labels, rotation=25, ha="right")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def run_safety_recovery_comparison(config: ProjectConfig, output: Path | None = None) -> Path:
    """Compare ordinary insertion with force-triggered stop-and-retract recovery."""
    directory = _output_directory(config, "safety_recovery", output)
    directory.mkdir(parents=True, exist_ok=True)
    trial_config = replace(
        config,
        output=replace(config.output, save_plot=False),
    )
    rows: list[dict[str, object]] = []
    for scenario, scenario_label, variation in SAFETY_RECOVERY_SCENARIOS:
        for controller_kind in ("impedance", "position"):
            for policy, guarded in (("unguarded", False), ("guarded", True)):
                summary, _ = run_peg(
                    trial_config,
                    viewer=False,
                    output=directory / scenario / controller_kind / policy,
                    variation=variation,
                    controller_kind=controller_kind,
                    force_safety=guarded,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario_label,
                        "policy": policy,
                        **asdict(variation),
                        **asdict(summary),
                    }
                )

    with (directory / "safety_recovery_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    guarded_rows = [row for row in rows if row["policy"] == "guarded"]
    unsafe_pairs: list[dict[str, object]] = []
    for guarded in guarded_rows:
        baseline = next(
            row
            for row in rows
            if row["scenario"] == guarded["scenario"]
            and row["controller"] == guarded["controller"]
            and row["policy"] == "unguarded"
        )
        if float(baseline["max_contact_force_n"]) > config.force_safety.force_limit_n:
            unsafe_pairs.append(
                {
                    "scenario": guarded["scenario"],
                    "controller": guarded["controller"],
                    "unguarded_peak_force_n": baseline["max_contact_force_n"],
                    "guarded_peak_force_n": guarded["max_contact_force_n"],
                    "unguarded_final_force_n": baseline["final_contact_force_n"],
                    "guarded_final_force_n": guarded["final_contact_force_n"],
                    "peak_force_reduction_n": float(baseline["max_contact_force_n"])
                    - float(guarded["max_contact_force_n"]),
                }
            )

    result = {
        "model_variant": config.model.variant,
        "force_safety": asdict(config.force_safety),
        "force_limit_is_exploratory": True,
        "taxonomy": {
            "success": "geometric completion without exceeding the force limit",
            "unsafe_completion": "geometric completion with an excessive peak force",
            "safety_abort_recovered": "force-triggered abort that released contact and retracted",
            "safety_abort": "force-triggered abort without verified release and retraction",
            "excessive_force": "failed insertion whose peak force exceeded the limit",
            "jammed": "failed insertion that retained contact without exceeding the force limit",
            "lateral_misalignment": (
                "failed insertion outside lateral clearance without retained contact"
            ),
            "insufficient_depth": "aligned insertion that did not reach the required depth",
        },
        "aggregates": {
            "evaluations": len(rows),
            "guarded_triggers": sum(bool(row["safety_triggered"]) for row in guarded_rows),
            "guarded_recoveries": sum(bool(row["safety_recovered"]) for row in guarded_rows),
            "unguarded_outcomes": dict(
                Counter(str(row["outcome"]) for row in rows if row["policy"] == "unguarded")
            ),
            "guarded_outcomes": dict(Counter(str(row["outcome"]) for row in guarded_rows)),
            "unsafe_pair_count": len(unsafe_pairs),
        },
        "unsafe_pairs": unsafe_pairs,
        "trials": rows,
    }
    with (directory / "safety_recovery_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if config.output.save_plot:
        _save_safety_recovery_plot(rows, config, directory / "safety_recovery_summary.png")
    return directory


def _save_contact_recovery_plot(
    rows: list[dict[str, object]], config: ProjectConfig, path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [
        f"{scenario_label}\n{controller}"
        for scenario, scenario_label, _ in SAFETY_RECOVERY_SCENARIOS
        for controller in ("impedance", "position")
    ]
    pairs = [
        {
            policy: next(
                row
                for row in rows
                if row["scenario"] == scenario
                and row["controller"] == controller
                and row["policy"] == policy
            )
            for policy in ("unguarded", "layered")
        }
        for scenario, _, _ in SAFETY_RECOVERY_SCENARIOS
        for controller in ("impedance", "position")
    ]
    indices = np.arange(len(pairs))
    width = 0.36
    colors = {"unguarded": "#b2182b", "layered": "#2166ac"}
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))

    for offset, policy in ((-width / 2, "unguarded"), (width / 2, "layered")):
        axes[0, 0].bar(
            indices + offset,
            [max(0.1, float(pair[policy]["max_contact_force_n"])) for pair in pairs],
            width,
            color=colors[policy],
            label=policy,
        )
        axes[0, 1].bar(
            indices + offset,
            [int(bool(pair[policy]["success"])) for pair in pairs],
            width,
            color=colors[policy],
            label=policy,
        )

    layered = [pair["layered"] for pair in pairs]
    recovery_codes = [
        3
        if bool(row["retry_success"])
        else 2
        if bool(row["safety_recovered"])
        else 1
        if bool(row["contact_recovery_triggered"])
        else 0
        for row in layered
    ]
    recovery_colors = ["#bdbdbd", "#fdae61", "#2b8cbe", "#1b7837"]
    axes[1, 0].bar(
        indices, recovery_codes, color=[recovery_colors[value] for value in recovery_codes]
    )

    policies = ("unguarded", "layered")
    counts = {
        policy: Counter(str(row["outcome"]) for row in rows if row["policy"] == policy)
        for policy in policies
    }
    outcomes = sorted({outcome for values in counts.values() for outcome in values})
    outcome_indices = np.arange(len(outcomes))
    for offset, policy in ((-width / 2, "unguarded"), (width / 2, "layered")):
        axes[1, 1].bar(
            outcome_indices + offset,
            [counts[policy].get(outcome, 0) for outcome in outcomes],
            width,
            color=colors[policy],
            label=policy,
        )

    axes[0, 0].axhline(
        config.force_safety.force_limit_n,
        color="#333333",
        linestyle="--",
        label=f"force abort ({config.force_safety.force_limit_n:g} N)",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Peak contact force [N]")
    axes[0, 0].set_title("Peak load")
    axes[0, 1].set_ylabel("Task success")
    axes[0, 1].set_yticks([0, 1], labels=["fail", "pass"])
    axes[0, 1].set_ylim(0.0, 1.15)
    axes[0, 1].set_title("Insertion result")
    axes[1, 0].set_ylabel("Recovery result")
    axes[1, 0].set_yticks(
        [0, 1, 2, 3],
        labels=["none", "spiral failed", "force retreat", "spiral retry success"],
    )
    axes[1, 0].set_ylim(0.0, 3.3)
    axes[1, 0].set_title("Layered recovery path")
    axes[1, 1].set_ylabel("Trials")
    axes[1, 1].set_title("Outcome taxonomy")
    axes[1, 1].set_xticks(outcome_indices, labels=outcomes, rotation=25, ha="right")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(indices, labels=labels, rotation=25, ha="right")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        if axis is not axes[1, 0]:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _contact_recovery_acceptance(
    rows: list[dict[str, object]], config: ProjectConfig
) -> dict[str, object]:
    """Evaluate the fixed recovery matrix against its regression expectations."""
    expected_keys = {
        (scenario, controller, policy)
        for scenario, _, _ in SAFETY_RECOVERY_SCENARIOS
        for controller in ("impedance", "position")
        for policy in ("unguarded", "layered")
    }
    keyed = {
        (str(row["scenario"]), str(row["controller"]), str(row["policy"])): row for row in rows
    }
    layered = [row for row in rows if row["policy"] == "layered"]
    nominal = [row for row in layered if row["scenario"] == "nominal"]
    retry = keyed.get(("x_plus_6mm", "impedance", "layered"))
    safety_aborts = [row for row in layered if bool(row["safety_triggered"])]
    unresolved_outcomes = {"unsafe_completion", "excessive_force", "jammed"}

    checks = {
        "complete_matrix": set(keyed) == expected_keys and len(rows) == len(expected_keys),
        "nominal_success_preserved": len(nominal) == 2
        and all(
            bool(row["success"])
            and not bool(row["contact_recovery_triggered"])
            and not bool(row["safety_triggered"])
            for row in nominal
        ),
        "known_jam_retry_succeeds": retry is not None
        and bool(retry["success"])
        and bool(retry["contact_recovery_triggered"])
        and bool(retry["spiral_entry_detected"])
        and bool(retry["retry_success"])
        and retry["outcome"] == "success_after_recovery",
        "retry_stays_below_force_limit": retry is not None
        and float(retry["max_contact_force_n"]) <= config.force_safety.force_limit_n,
        "high_force_path_is_exercised": bool(safety_aborts),
        "all_safety_aborts_release_and_retract": bool(safety_aborts)
        and all(
            bool(row["safety_recovered"])
            and float(row["final_contact_force_n"]) <= config.force_safety.release_force_n
            and float(row["retraction_distance_m"]) >= config.force_safety.minimum_retraction_m
            for row in safety_aborts
        ),
        "no_unresolved_layered_outcome": not any(
            str(row["outcome"]) in unresolved_outcomes for row in layered
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_contact_recovery_comparison(config: ProjectConfig, output: Path | None = None) -> Path:
    """Evaluate speed limiting, jam detection, spiral search, and force retreat together."""
    directory = _output_directory(config, "contact_recovery", output)
    directory.mkdir(parents=True, exist_ok=True)
    trial_config = replace(config, output=replace(config.output, save_plot=False))
    rows: list[dict[str, object]] = []
    for scenario, scenario_label, variation in SAFETY_RECOVERY_SCENARIOS:
        for controller_kind in ("impedance", "position"):
            for policy, layered in (("unguarded", False), ("layered", True)):
                summary, _ = run_peg(
                    trial_config,
                    viewer=False,
                    output=directory / scenario / controller_kind / policy,
                    variation=variation,
                    controller_kind=controller_kind,
                    force_safety=layered,
                    contact_recovery=layered,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario_label,
                        "policy": policy,
                        **asdict(variation),
                        **asdict(summary),
                    }
                )

    with (directory / "contact_recovery_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    layered_rows = [row for row in rows if row["policy"] == "layered"]
    result = {
        "model_variant": config.model.variant,
        "force_safety": asdict(config.force_safety),
        "contact_recovery": asdict(config.contact_recovery),
        "force_limits_are_exploratory": True,
        "policies": {
            "unguarded": "original time-parameterized insertion without recovery",
            "layered": (
                "axial speed limiting, low-force jam timeout, local retract and spiral retry, "
                "plus high-force stop and retreat"
            ),
        },
        "acceptance": _contact_recovery_acceptance(rows, config),
        "aggregates": {
            "evaluations": len(rows),
            "layered_successes": sum(bool(row["success"]) for row in layered_rows),
            "layered_force_recoveries": sum(bool(row["safety_recovered"]) for row in layered_rows),
            "layered_jam_detections": sum(
                bool(row["contact_recovery_triggered"]) for row in layered_rows
            ),
            "layered_spiral_entries": sum(
                bool(row["spiral_entry_detected"]) for row in layered_rows
            ),
            "layered_retry_successes": sum(bool(row["retry_success"]) for row in layered_rows),
            "unguarded_outcomes": dict(
                Counter(str(row["outcome"]) for row in rows if row["policy"] == "unguarded")
            ),
            "layered_outcomes": dict(Counter(str(row["outcome"]) for row in layered_rows)),
        },
        "trials": rows,
    }
    with (directory / "contact_recovery_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if config.output.save_plot:
        _save_contact_recovery_plot(rows, config, directory / "contact_recovery_summary.png")
    return directory


def _predictive_impact_acceptance(
    rows: list[dict[str, object]], config: ProjectConfig
) -> dict[str, object]:
    expected_keys = {
        (scenario, controller, policy)
        for scenario, _, _ in SAFETY_RECOVERY_SCENARIOS
        for controller in ("impedance", "position")
        for policy in ("reactive", "predictive")
    }
    keyed = {
        (str(row["scenario"]), str(row["controller"]), str(row["policy"])): row for row in rows
    }
    predictive = [row for row in rows if row["policy"] == "predictive"]
    nominal = [row for row in predictive if row["scenario"] == "nominal"]
    hazards = [row for row in predictive if row["scenario"] != "nominal"]
    reactive_diagonal = keyed.get(("xy_plus_8mm", "impedance", "reactive"))
    predictive_diagonal = keyed.get(("xy_plus_8mm", "impedance", "predictive"))
    predictive_jam = keyed.get(("x_plus_6mm", "impedance", "predictive"))
    safety_aborts = [row for row in predictive if bool(row["safety_triggered"])]
    unresolved = {"unsafe_completion", "excessive_force", "jammed", "recovery_failed"}
    checks = {
        "complete_matrix": set(keyed) == expected_keys and len(rows) == len(expected_keys),
        "nominal_success_preserved": len(nominal) == 2
        and all(bool(row["success"]) for row in nominal),
        "no_nominal_predictive_trigger": len(nominal) == 2
        and not any(bool(row["predictive_guard_triggered"]) for row in nominal),
        "hazard_prediction_exercised": any(
            bool(row["predictive_guard_triggered"]) for row in hazards
        ),
        "known_jam_remains_recoverable": predictive_jam is not None
        and bool(predictive_jam["success"]),
        "diagonal_impedance_peak_reduced_by_80_percent": reactive_diagonal is not None
        and predictive_diagonal is not None
        and float(predictive_diagonal["max_contact_force_n"])
        <= 0.20 * float(reactive_diagonal["max_contact_force_n"]),
        "diagonal_impedance_below_force_limit": predictive_diagonal is not None
        and float(predictive_diagonal["max_contact_force_n"]) <= config.force_safety.force_limit_n,
        "all_predictive_safety_aborts_recover": all(
            bool(row["safety_recovered"])
            and float(row["final_contact_force_n"]) <= config.force_safety.release_force_n
            for row in safety_aborts
        ),
        "no_unresolved_predictive_outcome": not any(
            str(row["outcome"]) in unresolved for row in predictive
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _save_predictive_impact_plot(
    rows: list[dict[str, object]], config: ProjectConfig, path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    pairs = []
    labels = []
    for scenario, scenario_label, _ in SAFETY_RECOVERY_SCENARIOS:
        for controller in ("impedance", "position"):
            pairs.append(
                {
                    policy: next(
                        row
                        for row in rows
                        if row["scenario"] == scenario
                        and row["controller"] == controller
                        and row["policy"] == policy
                    )
                    for policy in ("reactive", "predictive")
                }
            )
            labels.append(f"{scenario_label}\n{controller}")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    indices = np.arange(len(pairs))
    width = 0.36
    colors = {"reactive": "#b2182b", "predictive": "#2166ac"}
    for offset, policy in ((-width / 2, "reactive"), (width / 2, "predictive")):
        axes[0, 0].bar(
            indices + offset,
            [max(0.1, float(pair[policy]["max_contact_force_n"])) for pair in pairs],
            width,
            color=colors[policy],
            label=policy,
        )
        axes[0, 1].bar(
            indices + offset,
            [int(bool(pair[policy]["success"])) for pair in pairs],
            width,
            color=colors[policy],
            label=policy,
        )

    predictive_rows = [pair["predictive"] for pair in pairs]
    trigger_distances = [
        (
            1000.0 * float(row["predictive_trigger_distance_m"])
            if row["predictive_trigger_distance_m"] is not None
            else 0.0
        )
        for row in predictive_rows
    ]
    axes[1, 0].bar(
        indices,
        trigger_distances,
        color=[
            "#2b8cbe" if bool(row["predictive_guard_triggered"]) else "#bdbdbd"
            for row in predictive_rows
        ],
    )

    counts = {
        policy: Counter(str(row["outcome"]) for row in rows if row["policy"] == policy)
        for policy in ("reactive", "predictive")
    }
    outcomes = sorted({outcome for values in counts.values() for outcome in values})
    outcome_indices = np.arange(len(outcomes))
    for offset, policy in ((-width / 2, "reactive"), (width / 2, "predictive")):
        axes[1, 1].bar(
            outcome_indices + offset,
            [counts[policy].get(outcome, 0) for outcome in outcomes],
            width,
            color=colors[policy],
            label=policy,
        )

    axes[0, 0].axhline(
        config.force_safety.force_limit_n,
        color="#333333",
        linestyle="--",
        label=f"force abort ({config.force_safety.force_limit_n:g} N)",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Peak contact force [N]")
    axes[0, 0].set_title("Reactive vs predictive peak load")
    axes[0, 1].set_ylabel("Task success")
    axes[0, 1].set_yticks([0, 1], labels=["fail", "pass"])
    axes[0, 1].set_ylim(0.0, 1.15)
    axes[0, 1].set_title("Insertion result")
    axes[1, 0].axhline(
        1000.0 * config.predictive_impact.trigger_distance_m,
        color="#333333",
        linestyle="--",
        label="configured distance trigger",
    )
    axes[1, 0].set_ylabel("Trigger distance [mm]")
    axes[1, 0].set_title("Predictive intervention")
    axes[1, 1].set_ylabel("Trials")
    axes[1, 1].set_title("Outcome taxonomy")
    axes[1, 1].set_xticks(outcome_indices, labels=outcomes, rotation=25, ha="right")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(indices, labels=labels, rotation=25, ha="right")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def run_predictive_impact_comparison(config: ProjectConfig, output: Path | None = None) -> Path:
    """Compare reactive recovery with proximity- and force-rate-triggered intervention."""
    directory = _output_directory(config, "predictive_impact", output)
    directory.mkdir(parents=True, exist_ok=True)
    trial_config = replace(config, output=replace(config.output, save_plot=False))
    rows: list[dict[str, object]] = []
    for scenario, scenario_label, variation in SAFETY_RECOVERY_SCENARIOS:
        for controller_kind in ("impedance", "position"):
            for policy, predictive in (("reactive", False), ("predictive", True)):
                summary, _ = run_peg(
                    trial_config,
                    viewer=False,
                    output=directory / scenario / controller_kind / policy,
                    variation=variation,
                    controller_kind=controller_kind,
                    force_safety=True,
                    contact_recovery=True,
                    predictive_impact=predictive,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario_label,
                        "policy": policy,
                        **asdict(variation),
                        **asdict(summary),
                    }
                )

    with (directory / "predictive_impact_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    predictive_rows = [row for row in rows if row["policy"] == "predictive"]
    result = {
        "model_variant": config.model.variant,
        "force_safety": asdict(config.force_safety),
        "contact_recovery": asdict(config.contact_recovery),
        "predictive_impact": asdict(config.predictive_impact),
        "simulation_only_proximity_signal": True,
        "acceptance": _predictive_impact_acceptance(rows, config),
        "aggregates": {
            "evaluations": len(rows),
            "predictive_triggers": sum(
                bool(row["predictive_guard_triggered"]) for row in predictive_rows
            ),
            "predictive_successes": sum(bool(row["success"]) for row in predictive_rows),
            "predictive_safety_recoveries": sum(
                bool(row["safety_recovered"]) for row in predictive_rows
            ),
            "reactive_outcomes": dict(
                Counter(str(row["outcome"]) for row in rows if row["policy"] == "reactive")
            ),
            "predictive_outcomes": dict(Counter(str(row["outcome"]) for row in predictive_rows)),
        },
        "trials": rows,
    }
    with (directory / "predictive_impact_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    if config.output.save_plot:
        _save_predictive_impact_plot(rows, config, directory / "predictive_impact_summary.png")
    return directory
