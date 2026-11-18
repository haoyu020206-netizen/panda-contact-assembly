from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class ModelConfig:
    variant: str


@dataclass(frozen=True)
class SimulationConfig:
    timestep: float
    duration: float
    render_hz: float


@dataclass(frozen=True)
class JointControllerConfig:
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    move_duration: float
    target_offset: tuple[float, ...]


@dataclass(frozen=True)
class CartesianControllerConfig:
    position_stiffness: tuple[float, float, float]
    position_damping: tuple[float, float, float]
    orientation_stiffness: tuple[float, float, float]
    orientation_damping: tuple[float, float, float]
    nullspace_stiffness: float
    nullspace_damping: float
    damping_ratio: float
    target_offset: tuple[float, float, float]


@dataclass(frozen=True)
class CartesianPositionControllerConfig:
    joint_kp: tuple[float, ...]
    joint_kd: tuple[float, ...]
    translation_gain: float
    rotation_gain: float
    damping_ratio: float
    max_joint_velocity: float


@dataclass(frozen=True)
class PegTaskConfig:
    approach_time: float
    align_time: float
    insert_time: float
    settle_time: float
    insertion_depth: float
    target_depth: float
    clearance: float
    nominal_friction: float


@dataclass(frozen=True)
class MonteCarloConfig:
    seed: int
    trials: int
    lateral_offset_limit_m: float
    friction_scale_min: float
    friction_scale_max: float
    tool_mass_scale_min: float
    tool_mass_scale_max: float


@dataclass(frozen=True)
class BoundaryScanConfig:
    max_offset_m: float
    step_m: float


@dataclass(frozen=True)
class BoundaryRefinementConfig:
    min_offset_m: float
    max_offset_m: float
    step_m: float
    safe_force_limit_n: float


@dataclass(frozen=True)
class ForceSafetyConfig:
    force_limit_n: float
    trigger_duration_s: float
    hold_time_s: float
    retract_distance_m: float
    retract_duration_s: float
    release_force_n: float
    minimum_retraction_m: float


@dataclass(frozen=True)
class ContactRecoveryConfig:
    precontact_distance_m: float
    slow_zone_depth_m: float
    free_speed_m_s: float
    precontact_speed_m_s: float
    contact_slow_force_n: float
    contact_speed_m_s: float
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


@dataclass(frozen=True)
class PredictiveImpactConfig:
    distance_cutoff_m: float
    slowdown_distance_m: float
    trigger_distance_m: float
    minimum_closing_speed_m_s: float
    minimum_approach_axis_component: float
    probe_speed_m_s: float
    force_rate_limit_n_s: float
    force_rate_min_force_n: float
    lateral_speed_limit_m_s: float


@dataclass(frozen=True)
class OutputConfig:
    root: str
    save_csv: bool
    save_plot: bool


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    simulation: SimulationConfig
    joint_controller: JointControllerConfig
    cartesian_controller: CartesianControllerConfig
    cartesian_position_controller: CartesianPositionControllerConfig
    peg_task: PegTaskConfig
    monte_carlo: MonteCarloConfig
    boundary_scan: BoundaryScanConfig
    boundary_refinement: BoundaryRefinementConfig
    force_safety: ForceSafetyConfig
    contact_recovery: ContactRecoveryConfig
    predictive_impact: PredictiveImpactConfig
    output: OutputConfig


def _tuple(values: list[float], size: int, field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{field} must contain {size} values, got {len(result)}")
    return result


def _validate(raw: dict[str, Any]) -> ProjectConfig:
    model = raw.get("model", {"variant": "mvp"})
    simulation = raw["simulation"]
    joint = raw["joint_controller"]
    cartesian = raw["cartesian_controller"]
    cartesian_position = raw["cartesian_position_controller"]
    peg = raw["peg_task"]
    monte_carlo = raw["monte_carlo"]
    boundary_scan = raw["boundary_scan"]
    boundary_refinement = raw["boundary_refinement"]
    force_safety = raw["force_safety"]
    contact_recovery = raw["contact_recovery"]
    predictive_impact = raw["predictive_impact"]
    output = raw["output"]

    config = ProjectConfig(
        model=ModelConfig(variant=str(model["variant"])),
        simulation=SimulationConfig(
            timestep=float(simulation["timestep"]),
            duration=float(simulation["duration"]),
            render_hz=float(simulation["render_hz"]),
        ),
        joint_controller=JointControllerConfig(
            kp=_tuple(joint["kp"], 7, "joint_controller.kp"),
            kd=_tuple(joint["kd"], 7, "joint_controller.kd"),
            move_duration=float(joint["move_duration"]),
            target_offset=_tuple(joint["target_offset"], 7, "joint_controller.target_offset"),
        ),
        cartesian_controller=CartesianControllerConfig(
            position_stiffness=_tuple(
                cartesian["position_stiffness"], 3, "cartesian_controller.position_stiffness"
            ),
            position_damping=_tuple(
                cartesian["position_damping"], 3, "cartesian_controller.position_damping"
            ),
            orientation_stiffness=_tuple(
                cartesian["orientation_stiffness"],
                3,
                "cartesian_controller.orientation_stiffness",
            ),
            orientation_damping=_tuple(
                cartesian["orientation_damping"],
                3,
                "cartesian_controller.orientation_damping",
            ),
            nullspace_stiffness=float(cartesian["nullspace_stiffness"]),
            nullspace_damping=float(cartesian["nullspace_damping"]),
            damping_ratio=float(cartesian["damping_ratio"]),
            target_offset=_tuple(
                cartesian["target_offset"], 3, "cartesian_controller.target_offset"
            ),
        ),
        cartesian_position_controller=CartesianPositionControllerConfig(
            joint_kp=_tuple(
                cartesian_position["joint_kp"],
                7,
                "cartesian_position_controller.joint_kp",
            ),
            joint_kd=_tuple(
                cartesian_position["joint_kd"],
                7,
                "cartesian_position_controller.joint_kd",
            ),
            translation_gain=float(cartesian_position["translation_gain"]),
            rotation_gain=float(cartesian_position["rotation_gain"]),
            damping_ratio=float(cartesian_position["damping_ratio"]),
            max_joint_velocity=float(cartesian_position["max_joint_velocity"]),
        ),
        peg_task=PegTaskConfig(
            approach_time=float(peg["approach_time"]),
            align_time=float(peg["align_time"]),
            insert_time=float(peg["insert_time"]),
            settle_time=float(peg["settle_time"]),
            insertion_depth=float(peg["insertion_depth"]),
            target_depth=float(peg["target_depth"]),
            clearance=float(peg["clearance"]),
            nominal_friction=float(peg["nominal_friction"]),
        ),
        monte_carlo=MonteCarloConfig(
            seed=int(monte_carlo["seed"]),
            trials=int(monte_carlo["trials"]),
            lateral_offset_limit_m=float(monte_carlo["lateral_offset_limit_m"]),
            friction_scale_min=float(monte_carlo["friction_scale_min"]),
            friction_scale_max=float(monte_carlo["friction_scale_max"]),
            tool_mass_scale_min=float(monte_carlo["tool_mass_scale_min"]),
            tool_mass_scale_max=float(monte_carlo["tool_mass_scale_max"]),
        ),
        boundary_scan=BoundaryScanConfig(
            max_offset_m=float(boundary_scan["max_offset_m"]),
            step_m=float(boundary_scan["step_m"]),
        ),
        boundary_refinement=BoundaryRefinementConfig(
            min_offset_m=float(boundary_refinement["min_offset_m"]),
            max_offset_m=float(boundary_refinement["max_offset_m"]),
            step_m=float(boundary_refinement["step_m"]),
            safe_force_limit_n=float(boundary_refinement["safe_force_limit_n"]),
        ),
        force_safety=ForceSafetyConfig(
            force_limit_n=float(force_safety["force_limit_n"]),
            trigger_duration_s=float(force_safety["trigger_duration_s"]),
            hold_time_s=float(force_safety["hold_time_s"]),
            retract_distance_m=float(force_safety["retract_distance_m"]),
            retract_duration_s=float(force_safety["retract_duration_s"]),
            release_force_n=float(force_safety["release_force_n"]),
            minimum_retraction_m=float(force_safety["minimum_retraction_m"]),
        ),
        contact_recovery=ContactRecoveryConfig(
            precontact_distance_m=float(contact_recovery["precontact_distance_m"]),
            slow_zone_depth_m=float(contact_recovery["slow_zone_depth_m"]),
            free_speed_m_s=float(contact_recovery["free_speed_m_s"]),
            precontact_speed_m_s=float(contact_recovery["precontact_speed_m_s"]),
            contact_slow_force_n=float(contact_recovery["contact_slow_force_n"]),
            contact_speed_m_s=float(contact_recovery["contact_speed_m_s"]),
            jam_force_n=float(contact_recovery["jam_force_n"]),
            jam_timeout_s=float(contact_recovery["jam_timeout_s"]),
            jam_progress_m=float(contact_recovery["jam_progress_m"]),
            retract_distance_m=float(contact_recovery["retract_distance_m"]),
            retract_duration_s=float(contact_recovery["retract_duration_s"]),
            spiral_radius_m=float(contact_recovery["spiral_radius_m"]),
            spiral_cycles=int(contact_recovery["spiral_cycles"]),
            spiral_duration_s=float(contact_recovery["spiral_duration_s"]),
            spiral_descent_m=float(contact_recovery["spiral_descent_m"]),
            entry_depth_m=float(contact_recovery["entry_depth_m"]),
            retry_insert_duration_s=float(contact_recovery["retry_insert_duration_s"]),
            max_retries=int(contact_recovery["max_retries"]),
        ),
        predictive_impact=PredictiveImpactConfig(
            distance_cutoff_m=float(predictive_impact["distance_cutoff_m"]),
            slowdown_distance_m=float(predictive_impact["slowdown_distance_m"]),
            trigger_distance_m=float(predictive_impact["trigger_distance_m"]),
            minimum_closing_speed_m_s=float(predictive_impact["minimum_closing_speed_m_s"]),
            minimum_approach_axis_component=float(
                predictive_impact["minimum_approach_axis_component"]
            ),
            probe_speed_m_s=float(predictive_impact["probe_speed_m_s"]),
            force_rate_limit_n_s=float(predictive_impact["force_rate_limit_n_s"]),
            force_rate_min_force_n=float(predictive_impact["force_rate_min_force_n"]),
            lateral_speed_limit_m_s=float(predictive_impact["lateral_speed_limit_m_s"]),
        ),
        output=OutputConfig(
            root=str(output["root"]),
            save_csv=bool(output["save_csv"]),
            save_plot=bool(output["save_plot"]),
        ),
    )
    if config.simulation.timestep <= 0 or config.simulation.duration <= 0:
        raise ValueError("simulation timestep and duration must be positive")
    if config.model.variant not in {"mvp", "menagerie"}:
        raise ValueError("model.variant must be either 'mvp' or 'menagerie'")
    if config.cartesian_position_controller.max_joint_velocity <= 0:
        raise ValueError("cartesian_position_controller.max_joint_velocity must be positive")
    if config.monte_carlo.trials <= 0:
        raise ValueError("monte_carlo.trials must be positive")
    if config.monte_carlo.lateral_offset_limit_m <= 0:
        raise ValueError("monte_carlo.lateral_offset_limit_m must be positive")
    if config.monte_carlo.friction_scale_min > config.monte_carlo.friction_scale_max:
        raise ValueError("monte_carlo friction scale bounds are reversed")
    if config.monte_carlo.tool_mass_scale_min > config.monte_carlo.tool_mass_scale_max:
        raise ValueError("monte_carlo tool mass scale bounds are reversed")
    if config.boundary_scan.max_offset_m <= 0 or config.boundary_scan.step_m <= 0:
        raise ValueError("boundary_scan offsets must be positive")
    boundary_steps = config.boundary_scan.max_offset_m / config.boundary_scan.step_m
    if not abs(boundary_steps - round(boundary_steps)) < 1e-9:
        raise ValueError("boundary_scan.max_offset_m must be divisible by step_m")
    refinement = config.boundary_refinement
    if refinement.min_offset_m <= 0 or refinement.step_m <= 0:
        raise ValueError("boundary_refinement offsets must be positive")
    if refinement.max_offset_m < refinement.min_offset_m:
        raise ValueError("boundary_refinement offset bounds are reversed")
    refinement_steps = (refinement.max_offset_m - refinement.min_offset_m) / refinement.step_m
    if not abs(refinement_steps - round(refinement_steps)) < 1e-9:
        raise ValueError("boundary_refinement range must be divisible by step_m")
    if refinement.safe_force_limit_n <= 0:
        raise ValueError("boundary_refinement.safe_force_limit_n must be positive")
    safety = config.force_safety
    if safety.force_limit_n <= 0 or safety.release_force_n < 0:
        raise ValueError("force_safety force thresholds are invalid")
    if safety.trigger_duration_s < 0 or safety.hold_time_s < 0:
        raise ValueError("force_safety trigger and hold times cannot be negative")
    if safety.retract_distance_m <= 0 or safety.retract_duration_s <= 0:
        raise ValueError("force_safety retraction values must be positive")
    if not 0 < safety.minimum_retraction_m <= safety.retract_distance_m:
        raise ValueError("force_safety minimum_retraction_m must be within retract_distance_m")
    recovery = config.contact_recovery
    positive_recovery_values = (
        recovery.precontact_distance_m,
        recovery.slow_zone_depth_m,
        recovery.free_speed_m_s,
        recovery.precontact_speed_m_s,
        recovery.contact_slow_force_n,
        recovery.contact_speed_m_s,
        recovery.jam_force_n,
        recovery.jam_timeout_s,
        recovery.jam_progress_m,
        recovery.retract_distance_m,
        recovery.retract_duration_s,
        recovery.spiral_radius_m,
        recovery.spiral_duration_s,
        recovery.spiral_descent_m,
        recovery.entry_depth_m,
        recovery.retry_insert_duration_s,
    )
    if any(value <= 0 for value in positive_recovery_values):
        raise ValueError("contact_recovery values must be positive")
    if not recovery.contact_speed_m_s <= recovery.precontact_speed_m_s <= recovery.free_speed_m_s:
        raise ValueError("contact_recovery speeds must be ordered contact <= precontact <= free")
    if recovery.jam_force_n >= config.force_safety.force_limit_n:
        raise ValueError("contact_recovery.jam_force_n must be below force_safety.force_limit_n")
    if recovery.spiral_cycles <= 0 or recovery.max_retries <= 0:
        raise ValueError("contact_recovery cycles and retries must be positive")
    predictive = config.predictive_impact
    positive_predictive_values = (
        predictive.distance_cutoff_m,
        predictive.slowdown_distance_m,
        predictive.trigger_distance_m,
        predictive.minimum_closing_speed_m_s,
        predictive.minimum_approach_axis_component,
        predictive.probe_speed_m_s,
        predictive.force_rate_limit_n_s,
        predictive.force_rate_min_force_n,
        predictive.lateral_speed_limit_m_s,
    )
    if any(value <= 0 for value in positive_predictive_values):
        raise ValueError("predictive_impact values must be positive")
    if not (
        predictive.trigger_distance_m
        <= predictive.slowdown_distance_m
        < predictive.distance_cutoff_m
    ):
        raise ValueError("predictive_impact distances must satisfy trigger <= slowdown < cutoff")
    if predictive.probe_speed_m_s > recovery.precontact_speed_m_s:
        raise ValueError("predictive_impact.probe_speed_m_s cannot exceed precontact speed")
    if predictive.minimum_approach_axis_component > 1.0:
        raise ValueError("predictive minimum approach-axis component cannot exceed 1")
    if predictive.force_rate_min_force_n >= safety.force_limit_n:
        raise ValueError("predictive force-rate threshold must be below force safety limit")
    times = (
        config.peg_task.approach_time,
        config.peg_task.align_time,
        config.peg_task.insert_time,
        config.peg_task.settle_time,
    )
    if tuple(sorted(times)) != times:
        raise ValueError("peg task phase times must be monotonically increasing")
    if config.peg_task.target_depth < config.peg_task.insertion_depth:
        raise ValueError("peg_task.target_depth must be at least insertion_depth")
    return config


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return _validate(raw)
