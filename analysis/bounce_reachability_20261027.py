"""Passive 3D ball/floor trajectories copied from actual environment parameters."""
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'catch_it_copy'))
from gym_dcmm.envs.bounce_stable import make_bounce_stable
from ycb_objects.generalization_probe import physical_parameters


def words(a):
    return ' '.join(str(float(v)) for v in a)


def trajectory(env, speed_scale=1., distance_shift=0., timestep=None):
    base = env.unwrapped
    source = base.Dcmm.model
    p = physical_parameters(base)
    root = ET.Element('mujoco')
    ET.SubElement(root, 'option', timestep=str(timestep or source.opt.timestep), integrator='RK4')
    world = ET.SubElement(root, 'worldbody')
    floor = ET.SubElement(world, 'geom', name='floor', type='plane', size='10 10 .1')
    ball = ET.SubElement(world, 'body')
    ET.SubElement(ball, 'joint', type='free', damping=str(p['damping'][0]), armature=str(p['armature'][0]))
    ET.SubElement(ball, 'inertial', mass=str(p['mass']), pos=words(p['com']),
                  quat=words(p['inertia_quat']), diaginertia=words(p['inertia']))
    geom = ET.SubElement(ball, 'geom', name='ball', type='sphere', size=str(p['radius']))
    for node, gid in ((floor, base.floor_id), (geom, base.object_id)):
        node.set('solref', words(source.geom_solref[gid]))
        node.set('solimp', words(source.geom_solimp[gid]))
        node.set('friction', words(source.geom_friction[gid]))
        node.set('condim', str(source.geom_condim[gid]))
    pair = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_PAIR, 'bounce_v2_ground')
    if pair >= 0:
        contact = ET.SubElement(root, 'contact')
        ET.SubElement(contact, 'pair', geom1='floor', geom2='ball',
                      solref=words(source.pair_solref[pair]), solimp=words(source.pair_solimp[pair]),
                      friction=words(source.pair_friction[pair]), condim=str(source.pair_dim[pair]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    data.qpos[:3] = base.object_pos3d
    data.qpos[1] -= distance_shift
    data.qpos[3:] = base.object_q
    data.qvel[:] = base.object_vel6d
    data.qvel[:2] *= speed_scale
    data.qvel[3:5] *= speed_scale
    start = data.qpos[:3].copy()
    was_contact = False
    last_vz = data.qvel[2]
    impacts, apexes, samples = [], [], []
    count = int(2.5/model.opt.timestep)
    for i in range(count):
        mujoco.mj_step(model, data)
        touching = bool(data.ncon)
        if touching and not was_contact:
            impacts.append(dict(time=float(data.time), xyz=data.qpos[:3].tolist(),
                                speed_xy=float(np.linalg.norm(data.qvel[:2]))))
        if impacts and not touching and last_vz > 0 >= data.qvel[2]:
            if not apexes or data.time-apexes[-1]['time'] > .05:
                apexes.append(dict(time=float(data.time), xyz=data.qpos[:3].tolist(),
                                   speed_xy=float(np.linalg.norm(data.qvel[:2]))))
        if i % max(1, round(.1/model.opt.timestep)) == 0:
            samples.append(dict(time=float(data.time), xyz=data.qpos[:3].tolist()))
        was_contact, last_vz = touching, data.qvel[2]
    return dict(parameters=p, initial_xyz=start.tolist(), impacts=impacts[:5],
                apexes=apexes[:4], samples=samples, final_xyz=data.qpos[:3].tolist())


def main():
    rows = []
    for name in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
        env = make_bounce_stable(task='Catching', object_motion='bounce', test_object=name,
                                object_eval=False, viewer=False, render_mode=None)
        try:
            for seed in range(10):
                env.reset(seed=seed)
                for scale, shift in ((1., 0.), (2., 0.), (1., .8)):
                    result = trajectory(env, scale, shift)
                    rows.append(dict(object=name, seed=seed, speed_scale=scale,
                                     distance_shift=shift, **result))
        finally:
            env.close()
    output = ROOT/'analysis_artifacts/bounce_reachability_20261027.json'
    output.write_text(json.dumps(dict(assumptions='Passive isolated actual inertia/COM/contact/spin; no robot; launch ablations only, not a success benchmark', rows=rows), indent=2))
    for name in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
        for scale, shift in ((1., 0.), (2., 0.), (1., .8)):
            group = [r for r in rows if r['object']==name and r['speed_scale']==scale and r['distance_shift']==shift]
            peaks = [r['apexes'][0] for r in group if r['apexes']]
            print(name, scale, shift, 'first apex t/y/z mean', np.mean([[a['time'], a['xyz'][1], a['xyz'][2]] for a in peaks], axis=0),
                  'final y mean', np.mean([r['final_xyz'][1] for r in group]))


if __name__ == '__main__':
    main()
