#!/usr/bin/env python3
"""Read-only policy arm observation diagnostic: replay records or read live xArm.

Uses only MuJoCo forward kinematics; never steps physics or sends motion commands.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT/'catch_it_copy/assets/urdf/x1_xarm6_leap_right_object.xml'


class ArmObservation:
    def __init__(self, xml=DEFAULT_XML):
        self.source = Path(xml).resolve()
        original = ET.parse(self.source).getroot()
        base = original.find('./worldbody/body[@name="base_link"]')
        chain = base.find('body[@name="link1"]')
        arm_base = base.find('body[@name="arm_base"]')
        if chain is None or arm_base is None:
            raise ValueError('Expected the full project base_link, arm_base and link1 tree')
        # Preserve training geometry including its root tilt and mounting offsets.
        # Remove visual/collision meshes and hand so FK requires no renderer/assets.
        allowed = {'link{}'.format(i) for i in range(1,7)}
        def keep_body(body):
            result = ET.Element('body',dict(body.attrib))
            for child in body:
                if child.tag == 'joint' and child.get('name') in {'joint{}'.format(i) for i in range(1,7)}:
                    result.append(copy.deepcopy(child))
                elif child.tag == 'body' and child.get('name') in allowed:
                    result.append(keep_body(child))
            if result.find('joint') is not None:
                ET.SubElement(result,'inertial',pos='0 0 0',mass='1',diaginertia='1 1 1')
            return result
        kinematic = ET.Element('mujoco',model='policy_arm_readonly_fk')
        ET.SubElement(kinematic,'compiler',angle='radian',autolimits='true')
        world = ET.SubElement(kinematic,'worldbody')
        root = ET.SubElement(world,'body',dict(base.attrib))
        ET.SubElement(root,'body',dict(arm_base.attrib))
        root.append(keep_body(chain))
        self.kinematic_xml = ET.tostring(kinematic,encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.kinematic_xml)
        self.data = mujoco.MjData(self.model)
        self.ids = [mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'joint{}'.format(i)) for i in range(1,7)]
        self.addresses = [self.model.jnt_qposadr[i] for i in self.ids]
        self.base_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
        self.arm_base_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'arm_base')
        self.ee_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'link6')
        self.previous = None

    def convert(self, joints, timestamp):
        q = np.asarray(joints,dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or not math.isfinite(timestamp):
            raise ValueError('Need six finite actual joint angles in radians and a finite timestamp')
        self.data.qpos[self.addresses] = q
        mujoco.mj_fwdPosition(self.model,self.data)
        base_quat = self.data.xquat[self.base_id]
        # Exactly reproduce current env's planar-yaw position transform, rather
        # than replacing it with a full base rotation (these differ under tilt).
        yaw = 2*math.atan2(float(base_quat[3]),float(base_quat[0]))
        delta = self.data.xpos[self.ee_id]-self.data.xpos[self.arm_base_id]
        c,s = math.cos(yaw),math.sin(yaw)
        pos = np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],delta[2]])
        inverse = np.empty(4);quat = np.empty(4)
        mujoco.mju_negQuat(inverse,base_quat)
        mujoco.mju_mulQuat(quat,inverse,self.data.xquat[self.ee_id])
        velocity = np.zeros(3)
        dt = None if self.previous is None else timestamp-self.previous[0]
        valid_velocity = dt is not None and 0 < dt <= .2
        if dt is not None and dt <= 0:
            raise ValueError('Feedback timestamps must increase')
        if valid_velocity:
            velocity = (pos-self.previous[1])/dt
        self.previous = (timestamp,pos.copy())
        limits = self.model.jnt_range[self.ids]
        outside = [i+1 for i,jid in enumerate(self.ids)
                   if self.model.jnt_limited[jid] and not limits[i,0] <= q[i] <= limits[i,1]]
        return dict(arm_observation_10=np.r_[pos,quat,velocity].tolist(),
                    arm=dict(ee_pos3d=pos.tolist(),ee_quat=quat.tolist(),ee_v_lin_3d=velocity.tolist()),
                    velocity_valid=valid_velocity,dt_s=dt,
                    outside_training_joint_limits=outside,
                    hardware_model_alignment_verified=False,
                    note='Nominal training geometry; not a calibrated real-base transform. No robot commands.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--recording',type=Path,help='Directory containing states.jsonl (offline)')
    mode.add_argument('--live',action='store_true',help='Read robot feedback without sending motion commands')
    parser.add_argument('--ip',default='192.168.1.197')
    parser.add_argument('--duration',type=float,default=60)
    parser.add_argument('--hz',type=float,default=20)
    parser.add_argument('--xml',type=Path,default=DEFAULT_XML)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not math.isfinite(args.duration) or not math.isfinite(args.hz) or args.duration<=0 or not 0<args.hz<=50:
        parser.error('duration must be positive and hz in (0,50]')
    adapter=ArmObservation(args.xml)
    stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    output=args.output or ROOT/'analysis_artifacts'/('xarm_arm_obs_'+stamp)
    output.mkdir(parents=True,exist_ok=False)
    metadata=dict(mode='live' if args.live else 'offline',source_xml=str(adapter.source),
                  source_xml_sha256=hashlib.sha256(adapter.source.read_bytes()).hexdigest(),
                  observation_order=['x_m','y_m','z_m','qw','qx','qy','qz','vx_m_s','vy_m_s','vz_m_s'],
                  geometry='Full training XML arm chain including mounting offsets and base tilt; no added 2.4mm correction',
                  limitations='Not a physical base/encoder calibration. Velocity uses host joint-receive timestamps; first sample and gaps >0.2s invalid. '
                              'Arm-only observation: not a complete 24D policy input. Out-of-range joints are reported, never clipped or commanded.')
    arm=None;count=0;invalid=0;outside=set();last_print=0;complete=False
    def replay():
        with (args.recording/'states.jsonl').open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    def live():
        nonlocal arm
        from xarm.wrapper import XArmAPI
        from record_state import sample
        arm=XArmAPI(args.ip,is_radian=True)
        metadata.update(ip=args.ip,firmware=arm.version,tcp_offset_sdk_mm_rad=arm.tcp_offset,
                        world_offset_sdk_mm_rad=arm.world_offset)
        start=time.monotonic();deadline=start
        while time.monotonic()-start<args.duration:
            yield sample(arm)
            deadline=max(deadline+1/args.hz,time.monotonic())
            time.sleep(max(0,deadline-time.monotonic()))
    try:
        with (output/'observations.jsonl').open('x',buffering=1) as f:
            for raw in live() if args.live else replay():
                if raw.get('joint_query_is_real') is not True:
                    raise ValueError('Recording must explicitly contain actual joint feedback')
                result=adapter.convert(raw['joints_rad'],raw['host_joints_received'])
                result.update(index=count,host_joints_received=raw['host_joints_received'],
                              joints_rad=raw['joints_rad'],controller_state=raw['state'],error_warn=raw['error_warn'])
                f.write(json.dumps(result)+'\n');count+=1
                invalid+=not result['velocity_valid'];outside.update(result['outside_training_joint_limits'])
                if args.live and time.monotonic()-last_print>=1:
                    print('position_m={} quat_wxyz={} velocity_m_s={} outside_limits={}'.format(
                        np.round(result['arm']['ee_pos3d'],4),np.round(result['arm']['ee_quat'],4),
                        np.round(result['arm']['ee_v_lin_3d'],4),result['outside_training_joint_limits']),flush=True)
                    last_print=time.monotonic()
        complete=True
    finally:
        if arm is not None:arm.disconnect()
        (output/'kinematic_model.xml').write_text(adapter.kinematic_xml)
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        summary=dict(complete=complete,samples=count,invalid_velocity_samples=invalid,
                     outside_training_joint_limits=sorted(outside),motion_commands_sent=0)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('Saved:',output);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
