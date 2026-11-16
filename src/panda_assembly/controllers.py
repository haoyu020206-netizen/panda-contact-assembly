from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from panda_assembly.math_utils import damped_task_nullspace, rotation_error

FloatArray = NDArray[np.float64]


def site_pose(
    model: mujoco.MjModel, data: mujoco.MjData, site_id: int
) -> tuple[FloatArray, FloatArray]:
    position = data.site_xpos[site_id].copy()
    rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    return position, rotation


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> FloatArray:
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
    return np.vstack((jacobian_position, jacobian_rotation))


@dataclass
class JointPDController:
    kp: FloatArray
    kd: FloatArray

    def torque(
        self,
        data: mujoco.MjData,
        desired_position: FloatArray,
        desired_velocity: FloatArray,
    ) -> FloatArray:
        position_error = desired_position - data.qpos[:7]
        velocity_error = desired_velocity - data.qvel[:7]
        return self.kp * position_error + self.kd * velocity_error + data.qfrc_bias[:7]


@dataclass
class CartesianImpedanceController:
    position_stiffness: FloatArray
    position_damping: FloatArray
    orientation_stiffness: FloatArray
    orientation_damping: FloatArray
    nullspace_stiffness: float
    nullspace_damping: float
    damped_inverse_ratio: float
    rest_posture: FloatArray

    def torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_id: int,
        desired_position: FloatArray,
        desired_rotation: FloatArray,
        desired_linear_velocity: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        position, rotation = site_pose(model, data, site_id)
        jacobian = site_jacobian(model, data, site_id)
        twist = jacobian @ data.qvel
        target_velocity = (
            np.zeros(3) if desired_linear_velocity is None else desired_linear_velocity
        )

        position_error = desired_position - position
        orientation_delta = rotation_error(rotation, desired_rotation)
        wrench = np.concatenate(
            (
                self.position_stiffness * position_error
                + self.position_damping * (target_velocity - twist[:3]),
                self.orientation_stiffness * orientation_delta
                - self.orientation_damping * twist[3:],
            )
        )
        task_torque = jacobian.T @ wrench

        nullspace = damped_task_nullspace(jacobian, self.damped_inverse_ratio)
        posture_torque = (
            self.nullspace_stiffness * (self.rest_posture - data.qpos[:7])
            - self.nullspace_damping * data.qvel[:7]
        )
        torque = task_torque + nullspace @ posture_torque + data.qfrc_bias[:7]
        error = np.concatenate((position_error, orientation_delta))
        return torque, error


@dataclass
class CartesianPositionController:
    """Resolved-rate Cartesian position loop backed by a stiff joint servo."""

    joint_kp: FloatArray
    joint_kd: FloatArray
    translation_gain: float
    rotation_gain: float
    damped_inverse_ratio: float
    max_joint_velocity: float
    desired_joint_position: FloatArray

    def torque(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_id: int,
        desired_position: FloatArray,
        desired_rotation: FloatArray,
        desired_linear_velocity: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        position, rotation = site_pose(model, data, site_id)
        jacobian = site_jacobian(model, data, site_id)
        position_error = desired_position - position
        orientation_delta = rotation_error(rotation, desired_rotation)
        linear_feedforward = (
            np.zeros(3) if desired_linear_velocity is None else desired_linear_velocity
        )
        commanded_twist = np.concatenate(
            (
                linear_feedforward + self.translation_gain * position_error,
                self.rotation_gain * orientation_delta,
            )
        )
        task_identity = np.eye(jacobian.shape[0])
        joint_velocity = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T
            + self.damped_inverse_ratio * self.damped_inverse_ratio * task_identity,
            commanded_twist,
        )
        joint_velocity = np.clip(
            joint_velocity,
            -self.max_joint_velocity,
            self.max_joint_velocity,
        )
        self.desired_joint_position += model.opt.timestep * joint_velocity
        joint_limits = model.jnt_range[:7]
        self.desired_joint_position[:] = np.clip(
            self.desired_joint_position,
            joint_limits[:, 0] + 1e-4,
            joint_limits[:, 1] - 1e-4,
        )
        torque = (
            self.joint_kp * (self.desired_joint_position - data.qpos[:7])
            + self.joint_kd * (joint_velocity - data.qvel[:7])
            + data.qfrc_bias[:7]
        )
        error = np.concatenate((position_error, orientation_delta))
        return torque, error


def apply_torque(model: mujoco.MjModel, data: mujoco.MjData, torque: FloatArray) -> None:
    limits = model.actuator_ctrlrange[:7]
    data.ctrl[:7] = np.clip(torque, limits[:, 0], limits[:, 1])
