from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from panda_assembly.config import load_config
from panda_assembly.experiments import (
    run_cartesian,
    run_controller_boundary_refinement,
    run_controller_boundary_scan,
    run_controller_comparison,
    run_controller_robustness_comparison,
    run_joint,
    run_monte_carlo_comparison,
    run_peg,
    run_robustness_batch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panda-assembly",
        description="Run reproducible 7-DoF MuJoCo control experiments.",
    )
    parser.add_argument(
        "mode",
        choices=(
            "joint",
            "cartesian",
            "peg",
            "batch",
            "compare",
            "compare-batch",
            "monte-carlo",
            "boundary-scan",
            "boundary-refine",
        ),
        help="Experiment to run.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--controller",
        choices=("impedance", "position"),
        default="impedance",
        help="Cartesian controller used by cartesian and peg modes.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo's passive viewer (single runs only).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.mode == "batch":
        if args.viewer:
            raise SystemExit("--viewer is not available for batch runs")
        output = run_robustness_batch(config, args.output)
        print(json.dumps({"mode": "batch", "output": str(output)}, indent=2))
        return
    if args.mode == "compare":
        if args.viewer:
            raise SystemExit("--viewer is not available for comparison runs")
        output = run_controller_comparison(config, args.output)
        print(json.dumps({"mode": "compare", "output": str(output)}, indent=2))
        return
    if args.mode == "compare-batch":
        if args.viewer:
            raise SystemExit("--viewer is not available for comparison batches")
        output = run_controller_robustness_comparison(config, args.output)
        print(json.dumps({"mode": "compare-batch", "output": str(output)}, indent=2))
        return
    if args.mode == "monte-carlo":
        if args.viewer:
            raise SystemExit("--viewer is not available for Monte Carlo runs")
        output = run_monte_carlo_comparison(config, args.output)
        print(json.dumps({"mode": "monte-carlo", "output": str(output)}, indent=2))
        return
    if args.mode == "boundary-scan":
        if args.viewer:
            raise SystemExit("--viewer is not available for boundary scans")
        output = run_controller_boundary_scan(config, args.output)
        print(json.dumps({"mode": "boundary-scan", "output": str(output)}, indent=2))
        return
    if args.mode == "boundary-refine":
        if args.viewer:
            raise SystemExit("--viewer is not available for boundary refinement")
        output = run_controller_boundary_refinement(config, args.output)
        print(json.dumps({"mode": "boundary-refine", "output": str(output)}, indent=2))
        return

    if args.mode == "joint":
        summary, output = run_joint(config, viewer=args.viewer, output=args.output)
    elif args.mode == "cartesian":
        summary, output = run_cartesian(
            config,
            viewer=args.viewer,
            output=args.output,
            controller_kind=args.controller,
        )
    else:
        summary, output = run_peg(
            config,
            viewer=args.viewer,
            output=args.output,
            controller_kind=args.controller,
        )
    print(json.dumps({**asdict(summary), "output": str(output)}, indent=2))
