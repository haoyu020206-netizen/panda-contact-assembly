from __future__ import annotations

import json

import pytest

from scripts.aggregate_residual_ablation import (
    aggregate_runs,
    discover_runs,
    plot_summary,
    write_run_csv,
)


def write_run(root, seed: int, mode: str, learned_successes: int, learned_safety: int) -> None:
    directory = root / f"{mode}-{seed}"
    directory.mkdir()
    episodes = 10
    comparison = {
        "seed": seed,
        "rehearsal_mode": mode,
        "timesteps": 500,
        "evaluation_episodes": episodes,
        "non_regression": learned_successes >= 4 and learned_safety <= 4,
    }
    (directory / "comparison.json").write_text(json.dumps(comparison), encoding="utf-8")
    values = {
        "baseline": (4, 4, 180.0),
        "behavior-cloned": (8, 1, 20.0),
        "learned": (learned_successes, learned_safety, 10.0),
        "pose-reference": (10, 0, 5.0),
    }
    for name, (successes, safety, force) in values.items():
        policy_directory = directory / name
        policy_directory.mkdir()
        summary = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "safety_violations": safety,
            "mean_peak_force_n": force,
            "mean_reward": 0.0,
        }
        (policy_directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_aggregate_runs_separates_rehearsal_modes(tmp_path) -> None:
    write_run(tmp_path, 1, "none", learned_successes=3, learned_safety=5)
    write_run(tmp_path, 1, "teacher", learned_successes=10, learned_safety=0)
    write_run(tmp_path, 2, "none", learned_successes=5, learned_safety=3)
    write_run(tmp_path, 2, "teacher", learned_successes=9, learned_safety=1)

    aggregate = aggregate_runs(discover_runs(tmp_path))
    policies = {policy["policy"]: policy for policy in aggregate["policies"]}

    assert aggregate["seeds"] == [1, 2]
    assert aggregate["run_count"] == 4
    assert policies["baseline"]["episodes"] == 20
    assert policies["learned_none"]["success_rate"] == pytest.approx(0.4)
    assert policies["learned_none"]["safety_violations"] == 8
    assert policies["learned_teacher"]["success_rate"] == pytest.approx(0.95)
    assert policies["learned_teacher"]["safety_violations"] == 1

    csv_path = tmp_path / "ablation_runs.csv"
    plot_path = tmp_path / "ablation_summary.png"
    write_run_csv(discover_runs(tmp_path), csv_path)
    plot_summary(aggregate, plot_path)
    assert csv_path.stat().st_size > 0
    assert plot_path.stat().st_size > 0


def test_discover_runs_rejects_duplicate_identity(tmp_path) -> None:
    write_run(tmp_path, 1, "teacher", learned_successes=10, learned_safety=0)
    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    write_run(duplicate_root, 1, "teacher", learned_successes=10, learned_safety=0)

    with pytest.raises(ValueError, match="duplicate"):
        discover_runs(tmp_path)
