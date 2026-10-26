"""Controlled object/table diagnostic, not a policy success-rate evaluation."""
import argparse
import csv
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack

import mujoco
import numpy as np

from ycb_objects.loader import compose, load


def model_for(name, variant):
    resource = load(name)
    xml = compose('''<mujoco><option timestep="0.002" integrator="RK4"/>
      <worldbody><light pos="0 2 4"/><geom name="floor" type="plane" size="5 5 .1"/>
      <body name="table" pos="0 2.5 .68"><geom name="table_surface" type="box"
      size="1.2 1.6 .02" rgba=".55 .35 .15 1"/></body>
      <body name="object"><joint name="object" type="free" damping=".0001" armature=".001"/>
      <geom name="object" type="sphere" size=".04" friction="1.2 .1 .1"
      solref=".02 1" solimp=".9 .95 .01" condim="4"/></body></worldbody></mujoco>''', resource)
    model = mujoco.MjModel.from_xml_string(xml)
    bid = model.body('object').id
    if variant in ('no_damping', 'sphere_no_damping'):
        model.dof_damping[:] = 0
    if variant in ('sphere', 'sphere_no_damping'):
        root = ET.fromstring(xml)
        body = root.find('.//body[@name="object"]')
        for geom in body.findall('geom'):
            geom.set('contype', '1' if geom.get('name') == 'object' else '0')
            geom.set('conaffinity', '1' if geom.get('name') == 'object' else '0')
        ET.SubElement(body, 'inertial', mass=str(model.body_mass[bid]),
                      pos=' '.join(map(str, model.body_ipos[bid])),
                      quat=' '.join(map(str, model.body_iquat[bid])),
                      diaginertia=' '.join(map(str, model.body_inertia[bid])))
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        if variant == 'sphere_no_damping':
            model.dof_damping[:] = 0
    return model, resource[2]


def run(model, radius, speed, seconds, viewer=False):
    data = mujoco.MjData(model)
    data.qpos[:] = [0, 2.4, .7 + radius + .002, 1, 0, 0, 0]
    data.qvel[:] = [0, -speed, 0, speed / radius, 0, 0]
    mujoco.mj_forward(model, data)
    bid = model.body('object').id
    rows = []
    window = None
    stack = ExitStack()
    if viewer:
        from ycb_objects.viewer import passive_viewer
        window = stack.enter_context(passive_viewer(model, data))
        window.cam.lookat[:] = [0, 1.8, .7]
        window.cam.distance = 3
        window.cam.azimuth = 0
        window.cam.elevation = -30
    start = time.monotonic()
    try:
        for step in range(int(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if step % 10 == 0:
                pairs = [(int(c.geom1), int(c.geom2)) for c in data.contact
                         if bid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])]
                slip = []
                names = []
                for c in data.contact:
                    if bid not in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                        continue
                    jac = np.zeros((3, model.nv))
                    mujoco.mj_jac(model, data, jac, None, c.pos, bid)
                    velocity = jac @ data.qvel
                    normal = c.frame.reshape(3, 3)[0]
                    slip.append(float(np.linalg.norm(velocity - normal * np.dot(velocity, normal))))
                    names.append([model.geom(c.geom1).name, model.geom(c.geom2).name])
                rows.append(dict(t=float(data.time), x=float(data.qpos[0]), y=float(data.qpos[1]),
                                 z=float(data.qpos[2]), speed=float(np.linalg.norm(data.qvel[:3])),
                                 omega=float(np.linalg.norm(data.qvel[3:])), contacts=len(pairs),
                                 slip=max(slip, default=0.), pairs=str(names)))
                if window:
                    if not window.is_running():
                        break
                    window.sync()
                    time.sleep(max(0, start + data.time - time.monotonic()))
            if data.qpos[1] < .9 - radius:
                break
    finally:
        stack.close()
    return rows, dict(reached_edge=bool(data.qpos[1] < .9 - radius),
                      final_y=float(data.qpos[1]), final_speed=float(np.linalg.norm(data.qvel[:3])),
                      time=float(data.time), mass=float(model.body_mass[bid]),
                      inertia=model.body_inertia[bid].tolist(), com=model.body_ipos[bid].tolist(),
                      damping=model.dof_damping.tolist(), armature=model.dof_armature.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', default='tennis_ball')
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'no_damping', 'sphere', 'sphere_no_damping'],
                        default=['baseline', 'no_damping', 'sphere', 'sphere_no_damping'])
    parser.add_argument('--speeds', type=float, nargs='+', default=[.8, 1., 1.3])
    parser.add_argument('--seconds', type=float, default=8)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0 or any(s <= 0 for s in args.speeds):
        parser.error('Duration and speeds must be positive')
    if args.viewer and len(args.variants) * len(args.speeds) != 1:
        parser.error('Run one variant and one speed per viewer process')
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for variant in args.variants:
        for speed in args.speeds:
            model, radius = model_for(args.object, variant)
            rows, summary = run(model, radius, speed, args.seconds, args.viewer)
            key = f'{variant}_{speed:g}'
            with (args.output / (key + '.csv')).open('w') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            results[key] = summary
            (args.output / 'summary.json').write_text(json.dumps(results, indent=2))
            print(key, json.dumps(summary), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
