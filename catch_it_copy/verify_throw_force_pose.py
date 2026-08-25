"""Check the initial throw_force end-effector and ball pose."""

import numpy as np
import mujoco

import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM


def main():
    dcmm = MJ_DCMM(viewer=False, object_name="object", object_eval=False)
    dcmm.data.qpos[15:21] = DcmmCfg.throw_force_arm_joints
    dcmm.data_arm.qpos[0:6] = DcmmCfg.throw_force_arm_joints
    mujoco.mj_forward(dcmm.model, dcmm.data)
    mujoco.mj_forward(dcmm.model_arm, dcmm.data_arm)

    ee_xmat = dcmm.data.body("link6").xmat.reshape(3, 3)
    palm_normal = ee_xmat[:, 1]
    finger_direction = ee_xmat[:, 2]
    ee_xpos = dcmm.data.body("link6").xpos.copy()
    ball_offset = palm_normal * (DcmmCfg.throw_force_radius + 0.02)
    ball_pos = ee_xpos + ball_offset

    print("link6 palm normal:", np.round(palm_normal, 4))
    print("link6 finger direction:", np.round(finger_direction, 4))
    print("ball offset:", np.round(ball_offset, 4))
    print("ball distance:", np.linalg.norm(ball_offset))

    assert palm_normal[2] < -0.2, "palm normal is not pointing down"
    assert finger_direction[2] < -0.5, "finger direction is not pointing down"
    expected_distance = DcmmCfg.throw_force_radius + 0.02
    assert np.isclose(np.linalg.norm(ball_pos - ee_xpos), expected_distance), (
        "ball is not at the configured palm offset"
    )
    print("throw_force pose checks passed")


if __name__ == "__main__":
    main()