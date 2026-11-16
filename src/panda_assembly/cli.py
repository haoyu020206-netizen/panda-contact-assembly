from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from panda_assembly.config import load_config
from panda_assembly.experiments import (
    run_cartesian,
    run_joint,
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
        choices=("joint", "cartesian", "peg", "batch"),
        help="Experiment to run.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mvp.toml"))
    parser.add_argument("--output", type=Path, default=None)
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

    runners = {
        "joint": run_joint,
        "cartesian": run_cartesian,
        "peg": run_peg,
    }
    summary, output = runners[args.mode](config, viewer=args.viewer, output=args.output)
    print(json.dumps({**asdict(summary), "output": str(output)}, indent=2))
