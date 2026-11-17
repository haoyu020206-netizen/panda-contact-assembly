from pathlib import Path

import pytest

from panda_assembly.config import load_config

CONFIG = Path(__file__).parents[1] / "configs" / "mvp.toml"


def test_loads_mvp_config() -> None:
    config = load_config(CONFIG)
    assert config.model.variant == "mvp"
    assert config.simulation.timestep == pytest.approx(0.002)
    assert len(config.joint_controller.kp) == 7
    assert len(config.cartesian_position_controller.joint_kp) == 7
    assert config.peg_task.approach_time < config.peg_task.insert_time
    assert config.peg_task.target_depth >= config.peg_task.insertion_depth
    assert config.monte_carlo.trials == 20
    assert config.boundary_scan.max_offset_m == 0.012
    assert config.boundary_scan.step_m == 0.002
