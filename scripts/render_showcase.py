from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from panda_assembly.experiments import Variation, _configure_variation
from panda_assembly.model import load_model, reset_to_home


def final_joint_position(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return np.asarray([float(rows[-1][f"q{i}_rad"]) for i in range(1, 8)])


def render(timeseries: Path, output: Path) -> None:
    model = load_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    _, _, hole_id = _configure_variation(model, data, Variation(), nominal_friction=0.8)
    data.qpos[:7] = final_joint_position(timeseries)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = data.site_xpos[hole_id] + np.array([-0.10, 0.0, 0.20])
    camera.distance = 1.15
    camera.azimuth = 135.0
    camera.elevation = -24.0

    renderer = mujoco.Renderer(model, height=720, width=1280)
    try:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
    finally:
        renderer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the final verified peg-in-hole state.")
    parser.add_argument("--timeseries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.timeseries, args.output)


if __name__ == "__main__":
    main()
