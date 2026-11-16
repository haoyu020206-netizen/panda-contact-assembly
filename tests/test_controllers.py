import mujoco
import numpy as np

from panda_assembly.controllers import CartesianPositionController, site_pose
from panda_assembly.model import load_model, reset_to_home


def test_cartesian_position_controller_holds_current_pose() -> None:
    model = load_model()
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    position, rotation = site_pose(model, data, site_id)
    initial_posture = data.qpos[:7].copy()
    controller = CartesianPositionController(
        joint_kp=np.full(7, 100.0),
        joint_kd=np.full(7, 10.0),
        translation_gain=8.0,
        rotation_gain=6.0,
        damped_inverse_ratio=0.03,
        max_joint_velocity=1.5,
        desired_joint_position=initial_posture.copy(),
    )

    torque, error = controller.torque(model, data, site_id, position, rotation)

    np.testing.assert_allclose(error, 0.0, atol=1e-12)
    np.testing.assert_allclose(controller.desired_joint_position, initial_posture, atol=1e-12)
    np.testing.assert_allclose(torque, data.qfrc_bias[:7], atol=1e-12)
