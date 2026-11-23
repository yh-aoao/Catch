"""Offline adapter contract tests; no SDK or hardware connection."""
import copy
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from observation_adapter import ArmObservation, DEFAULT_XML, ROOT


class Tests(unittest.TestCase):
    def test_matches_full_training_body_tree(self):
        # Independent reference retains the entire base, wheels and hand tree.
        src=ET.parse(DEFAULT_XML).getroot()
        base=copy.deepcopy(src.find('./worldbody/body[@name="base_link"]'))
        for body in list(base.iter('body')):
            for child in list(body):
                if child.tag in ('geom','camera','light','site'):body.remove(child)
        reference=ET.Element('mujoco');ET.SubElement(reference,'compiler',angle='radian',autolimits='true')
        ET.SubElement(reference,'worldbody').append(base)
        model=mujoco.MjModel.from_xml_string(ET.tostring(reference,encoding='unicode'));data=mujoco.MjData(model)
        adapter=ArmObservation()
        rows=[json.loads(l) for l in (ROOT/'analysis_artifacts/xarm_readonly_20261123_152245/states.jsonl').read_text().splitlines()]
        b=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
        a=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'arm_base')
        ee=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link6')
        for index,row in enumerate(rows[::24]):
            for j,q in enumerate(row['joints_rad'],1):
                jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'joint'+str(j))
                data.qpos[model.jnt_qposadr[jid]]=q
            mujoco.mj_fwdPosition(model,data)
            quat=data.xquat[b];yaw=2*np.arctan2(quat[3],quat[0]);delta=data.xpos[ee]-data.xpos[a]
            pos=np.array([np.cos(yaw)*delta[0]+np.sin(yaw)*delta[1],-np.sin(yaw)*delta[0]+np.cos(yaw)*delta[1],delta[2]])
            orientation=Rotation.from_matrix(data.xmat[b].reshape(3,3)).inv()*Rotation.from_matrix(data.xmat[ee].reshape(3,3))
            result=adapter.convert(row['joints_rad'],index*.05)
            np.testing.assert_allclose(result['arm']['ee_pos3d'],pos,atol=1e-12)
            out=np.array(result['arm']['ee_quat'])
            self.assertLess((Rotation.from_quat(out[[1,2,3,0]]).inv()*orientation).magnitude(),1e-12)
            self.assertEqual(len(result['arm_observation_10']),10)

    def test_velocity_and_invalid_gap(self):
        adapter=ArmObservation();q=np.zeros(6)
        one=adapter.convert(q,10);self.assertFalse(one['velocity_valid'])
        q[0]=.01;two=adapter.convert(q,10.05)
        np.testing.assert_allclose(two['arm']['ee_v_lin_3d'],(np.array(two['arm']['ee_pos3d'])-one['arm']['ee_pos3d'])/.05)
        self.assertTrue(two['velocity_valid'])
        gap=adapter.convert(q,11);self.assertFalse(gap['velocity_valid'])
        np.testing.assert_array_equal(gap['arm']['ee_v_lin_3d'],[0,0,0])
        with self.assertRaises(ValueError):adapter.convert(q,11)

    def test_limits_are_reported_without_clipping(self):
        adapter=ArmObservation();q=np.zeros(6)
        out=adapter.convert(q,1)
        self.assertIn(4,out['outside_training_joint_limits']);self.assertIn(6,out['outside_training_joint_limits'])
        np.testing.assert_array_equal(adapter.data.qpos[adapter.addresses],q)
        with self.assertRaises(ValueError):adapter.convert([float('nan')]*6,2)


if __name__=='__main__':unittest.main()
