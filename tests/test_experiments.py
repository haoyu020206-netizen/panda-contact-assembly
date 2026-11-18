from dataclasses import replace
from pathlib import Path

from panda_assembly.config import load_config
from panda_assembly.experiments import (
    BOUNDARY_DIRECTIONS,
    ROBUSTNESS_SCENARIOS,
    SAFETY_RECOVERY_SCENARIOS,
    _boundary_refinement_points,
    _boundary_scan_points,
    _contact_recovery_acceptance,
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


def test_safety_recovery_scenarios_cover_nominal_and_known_hazards() -> None:
    names = [name for name, _, _ in SAFETY_RECOVERY_SCENARIOS]
    assert names == ["nominal", "x_plus_6mm", "xy_plus_8mm"]
    assert len(names) == len(set(names))


def test_contact_recovery_acceptance_requires_nominal_retry_and_safe_abort() -> None:
    config = load_config(CONFIG)
    rows = []
    for scenario, _, _ in SAFETY_RECOVERY_SCENARIOS:
        for controller in ("impedance", "position"):
            for policy in ("unguarded", "layered"):
                row = {
                    "scenario": scenario,
                    "controller": controller,
                    "policy": policy,
                    "success": scenario == "nominal",
                    "outcome": "success" if scenario == "nominal" else "safety_abort_recovered",
                    "max_contact_force_n": 100.0,
                    "final_contact_force_n": 0.0,
                    "safety_triggered": policy == "layered" and scenario == "xy_plus_8mm",
                    "safety_recovered": policy == "layered" and scenario == "xy_plus_8mm",
                    "retraction_distance_m": 0.08,
                    "contact_recovery_triggered": False,
                    "spiral_entry_detected": False,
                    "retry_success": False,
                }
                rows.append(row)

    retry = next(
        row
        for row in rows
        if (row["scenario"], row["controller"], row["policy"])
        == ("x_plus_6mm", "impedance", "layered")
    )
    retry.update(
        success=True,
        outcome="success_after_recovery",
        contact_recovery_triggered=True,
        spiral_entry_detected=True,
        retry_success=True,
    )

    acceptance = _contact_recovery_acceptance(rows, config)
    assert acceptance["passed"]
    retry["retry_success"] = False
    assert not _contact_recovery_acceptance(rows, config)["passed"]


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


def test_boundary_refinement_points_cover_transition_grid() -> None:
    config = load_config(CONFIG)
    points = _boundary_refinement_points(config)
    refinement = config.boundary_refinement
    fine_steps = round((refinement.max_offset_m - refinement.min_offset_m) / refinement.step_m)
    expected_radii = 1 + fine_steps + 1

    assert len(points) == 1 + expected_radii * len(BOUNDARY_DIRECTIONS)
    assert points[0].direction == "origin"
    positive_offsets = sorted({point.offset_m for point in points[1:]})
    assert positive_offsets[0] == config.boundary_scan.step_m
    assert positive_offsets[1] == refinement.min_offset_m
    assert positive_offsets[-1] == refinement.max_offset_m
    assert all(
        abs(
            (point.variation.lateral_offset_x**2 + point.variation.lateral_offset_y**2) ** 0.5
            - point.offset_m
        )
        < 1e-12
        for point in points
    )
