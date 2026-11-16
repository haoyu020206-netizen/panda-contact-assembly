import mujoco
import numpy as np

from panda_assembly.experiments import _site_offset_in_frame
from panda_assembly.model import load_model, model_path, reset_to_home


def test_mvp_model_compiles_and_has_expected_interfaces() -> None:
    model = load_model()
    assert model.nq == 7
    assert model.nv == 7
    assert model.nu == 7
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site") >= 0


def test_first_joint_responds_to_direct_motor_torque() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    initial_position = float(data.qpos[0])

    for _ in range(200):
        data.ctrl[:] = data.qfrc_bias[: model.nu]
        data.ctrl[0] += 10.0
        mujoco.mj_step(model, data)

    displacement = float(data.qpos[0] - initial_position)
    contacts = [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2),
            float(data.contact[i].dist),
        )
        for i in range(data.ncon)
    ]
    assert abs(displacement) > 0.01, (
        f"joint1 displacement={displacement}, contacts={contacts}, "
        f"constraint={np.array2string(data.qfrc_constraint, precision=3)}"
    )


def test_menagerie_model_compiles_with_same_control_interface() -> None:
    model = load_model(variant="menagerie")
    assert model.nq == 7
    assert model.nv == 7
    assert model.nu == 7
    assert model.body("link1").mass[0] > 4.0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hole_target") >= 0

    data = mujoco.MjData(model)
    reset_to_home(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
    ee_z_axis = data.site_xmat[ee_id].reshape(3, 3)[:, 2]
    assert ee_z_axis[2] > 0.99
    assert data.site_xpos[tip_id, 2] < data.site_xpos[ee_id, 2]


def test_tool_offset_is_frame_correct_for_both_model_variants() -> None:
    for variant in ("mvp", "menagerie"):
        model = load_model(variant=variant)
        data = mujoco.MjData(model)
        reset_to_home(model, data)
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip")
        offset = _site_offset_in_frame(model, data, ee_id, tip_id)
        np.testing.assert_allclose(offset, [0.0, 0.0, -0.13], atol=1e-9)


def test_model_variant_rejects_unknown_name() -> None:
    with np.testing.assert_raises_regex(ValueError, "Unknown model variant"):
        model_path("unknown")
