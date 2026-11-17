from dataclasses import replace
from pathlib import Path

from panda_assembly.config import load_config
from panda_assembly.experiments import (
    BOUNDARY_DIRECTIONS,
    ROBUSTNESS_SCENARIOS,
    _boundary_scan_points,
    _monte_carlo_variations,
    run_joint,
    wilson_score_interval,
)

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


def test_robustness_scenario_names_are_unique() -> None:
    names = [name for name, _ in ROBUSTNESS_SCENARIOS]
    assert len(names) == 8
    assert len(names) == len(set(names))
    assert names[0] == "nominal"


def test_monte_carlo_variations_are_reproducible_and_bounded() -> None:
    config = load_config(CONFIG)
    first = _monte_carlo_variations(config)
    second = _monte_carlo_variations(config)

    assert first == second
    assert len(first) == config.monte_carlo.trials
    assert all(
        abs(variation.lateral_offset_x) <= config.monte_carlo.lateral_offset_limit_m
        and abs(variation.lateral_offset_y) <= config.monte_carlo.lateral_offset_limit_m
        and config.monte_carlo.friction_scale_min
        <= variation.friction_scale
        <= config.monte_carlo.friction_scale_max
        and config.monte_carlo.tool_mass_scale_min
        <= variation.tool_mass_scale
        <= config.monte_carlo.tool_mass_scale_max
        for variation in first
    )


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_score_interval(12, 20)
    assert lower < 0.6 < upper
    assert 0.0 <= lower <= upper <= 1.0


def test_boundary_scan_points_are_unique_and_radial() -> None:
    config = load_config(CONFIG)
    points = _boundary_scan_points(config)
    steps = round(config.boundary_scan.max_offset_m / config.boundary_scan.step_m)

    assert len(points) == 1 + steps * len(BOUNDARY_DIRECTIONS)
    assert points[0].scenario == "origin"
    assert len({point.scenario for point in points}) == len(points)
    assert {point.direction for point in points[1:]} == {
        direction for direction, _, _, _, _ in BOUNDARY_DIRECTIONS
    }
    assert all(
        abs(
            (point.variation.lateral_offset_x**2 + point.variation.lateral_offset_y**2) ** 0.5
            - point.offset_m
        )
        < 1e-12
        for point in points
    )
