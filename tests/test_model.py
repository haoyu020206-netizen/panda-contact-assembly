import mujoco
import numpy as np

from panda_assembly.model import load_model, reset_to_home


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

    data.ctrl[0] = 10.0
    for _ in range(200):
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
