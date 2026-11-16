from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


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
class OutputConfig:
    root: str
    save_csv: bool
    save_plot: bool


@dataclass(frozen=True)
class ProjectConfig:
    simulation: SimulationConfig
    joint_controller: JointControllerConfig
    cartesian_controller: CartesianControllerConfig
    peg_task: PegTaskConfig
    output: OutputConfig


def _tuple(values: list[float], size: int, field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{field} must contain {size} values, got {len(result)}")
    return result


def _validate(raw: dict[str, Any]) -> ProjectConfig:
    simulation = raw["simulation"]
    joint = raw["joint_controller"]
    cartesian = raw["cartesian_controller"]
    peg = raw["peg_task"]
    output = raw["output"]

    config = ProjectConfig(
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
        output=OutputConfig(
            root=str(output["root"]),
            save_csv=bool(output["save_csv"]),
            save_plot=bool(output["save_plot"]),
        ),
    )
    if config.simulation.timestep <= 0 or config.simulation.duration <= 0:
        raise ValueError("simulation timestep and duration must be positive")
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
