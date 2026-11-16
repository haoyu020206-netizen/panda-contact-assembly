import mujoco

from panda_assembly.model import load_model


def test_mvp_model_compiles_and_has_expected_interfaces() -> None:
    model = load_model()
    assert model.nq == 7
    assert model.nv == 7
    assert model.nu == 7
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site") >= 0
