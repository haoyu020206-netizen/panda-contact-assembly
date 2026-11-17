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
