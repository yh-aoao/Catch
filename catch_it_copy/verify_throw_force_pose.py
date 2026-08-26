"""Check the initial throw_force end-effector and ball pose."""

import numpy as np
import mujoco

import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.agents.MujocoDcmm import MJ_DCMM


def main():
    dcmm = MJ_DCMM(viewer=False, object_name="object", object_eval=False)
    dcmm.data.qpos[15:21] = DcmmCfg.throw_force_arm_joints
    dcmm.data.qpos[21:37] = np.array([
        1.2, 0, 0.8, 0.8,
        1.2, 0, 0.8, 0.8,
        1.2, 0, 0.8, 0.8,
        0, 0.6, 0.6, 0.6,
    ])
    dcmm.data_arm.qpos[0:6] = DcmmCfg.throw_force_arm_joints
    mujoco.mj_forward(dcmm.model, dcmm.data)
    mujoco.mj_forward(dcmm.model_arm, dcmm.data_arm)

    ee_xpos = dcmm.data.body("link6").xpos.copy()
    mcp_center = np.mean(
        [dcmm.data.body(name).xpos for name in ("mcp_joint", "mcp_joint_2", "mcp_joint_3")],
        axis=0,
    )
    fingertip_center = np.mean(
        [dcmm.data.body(name).xpos for name in ("fingertip", "fingertip_2", "fingertip_3")],
        axis=0,
    )
    finger_span = fingertip_center - mcp_center
    ball_pos = mcp_center + 0.35 * finger_span

    structural_contacts = []
    for contact_index in range(dcmm.data.ncon):
        contact = dcmm.data.contact[contact_index]
        first_body = dcmm.model.geom_bodyid[contact.geom1]
        second_body = dcmm.model.geom_bodyid[contact.geom2]
        first_name = dcmm.model.body(first_body).name
        second_name = dcmm.model.body(second_body).name
        arm_bodies = {
            "base_link", "link1", "link2", "link3", "link4", "link5", "link6"
        }
        if first_name in arm_bodies and second_name in arm_bodies:
            structural_contacts.append((first_name, second_name, float(contact.dist)))

    print("finger root center:", np.round(mcp_center, 4))
    print("fingertip center:", np.round(fingertip_center, 4))
    print("ball position:", np.round(ball_pos, 4))
    print("arm structural contacts:", structural_contacts)

    assert finger_span[2] < -0.02, "fingertips are not below the finger roots"
    assert not structural_contacts, "arm links are colliding"
    expected_distance = DcmmCfg.throw_force_radius + 0.02
    ball_ratio = np.linalg.norm(ball_pos - mcp_center) / np.linalg.norm(finger_span)
    print("ball grasp ratio:", round(float(ball_ratio), 4))
    assert np.isclose(ball_ratio, 0.35, atol=1e-3), (
        "ball is not between the finger roots and fingertips"
    )
    print("throw_force pose checks passed")


if __name__ == "__main__":
    main()