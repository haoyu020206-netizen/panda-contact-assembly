from dataclasses import replace
from pathlib import Path

from panda_assembly.config import load_config
from panda_assembly.experiments import run_joint

CONFIG = Path(__file__).parents[1] / "configs" / "mvp.toml"


def test_joint_experiment_smoke_run(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config = replace(
        config,
        simulation=replace(config.simulation, duration=0.01),
        output=replace(config.output, save_csv=False, save_plot=False),
    )

    summary, output = run_joint(config, output=tmp_path / "joint")

    assert summary.mode == "joint"
    assert summary.rms_joint_error_rad is not None
    assert (output / "summary.json").is_file()
