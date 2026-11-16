from pathlib import Path

import pytest

from panda_assembly.config import load_config

CONFIG = Path(__file__).parents[1] / "configs" / "mvp.toml"


def test_loads_mvp_config() -> None:
    config = load_config(CONFIG)
    assert config.simulation.timestep == pytest.approx(0.002)
    assert len(config.joint_controller.kp) == 7
    assert config.peg_task.approach_time < config.peg_task.insert_time
    assert config.peg_task.target_depth >= config.peg_task.insertion_depth
