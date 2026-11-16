from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path

import mujoco

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"motor{i}" for i in range(1, 8))
MODEL_FILES = {
    "mvp": "panda_mvp.xml",
    "menagerie": "panda_menagerie.xml",
}


def model_path(variant: str = "mvp") -> Path:
    try:
        filename = MODEL_FILES[variant]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_FILES))
        raise ValueError(f"Unknown model variant {variant!r}; choose one of: {choices}") from error
    resource = files("panda_assembly").joinpath(f"assets/{filename}")
    with as_file(resource) as path:
        return Path(path)


def default_model_path() -> Path:
    return model_path("mvp")


def load_model(
    path: str | Path | None = None,
    timestep: float | None = None,
    variant: str = "mvp",
) -> mujoco.MjModel:
    selected_path = Path(path) if path is not None else model_path(variant)
    model = mujoco.MjModel.from_xml_path(str(selected_path))
    if timestep is not None:
        model.opt.timestep = timestep
    validate_model(model)
    return model


def validate_model(model: mujoco.MjModel) -> None:
    if model.nq != 7 or model.nv != 7:
        raise ValueError(f"Expected a fixed-base 7-DoF model, got nq={model.nq}, nv={model.nv}")
    if model.nu != 7:
        raise ValueError(f"Expected seven torque actuators, got nu={model.nu}")
    for name in (*ARM_JOINTS, *ARM_ACTUATORS, "ee_site", "peg_tip", "hole_target"):
        object_type = mujoco.mjtObj.mjOBJ_SITE
        if name.startswith("joint"):
            object_type = mujoco.mjtObj.mjOBJ_JOINT
        elif name.startswith("motor"):
            object_type = mujoco.mjtObj.mjOBJ_ACTUATOR
        if mujoco.mj_name2id(model, object_type, name) < 0:
            raise ValueError(f"Model is missing required object: {name}")


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
