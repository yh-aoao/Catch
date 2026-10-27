"""Isolated passive-drop timestep/impact-phase diagnostic, not material calibration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def drop(mass, radius, height, timestep, duration=1.5, time_constant=.008,
         damping_ratio=.1, joint_damping=.00015, armature=.0001, integrator='RK4',
         vertical_velocity=-.2):
    values = (mass, radius, height, timestep, duration, time_constant, damping_ratio)
    if any(not np.isfinite(v) or v <= 0 for v in values) or height <= radius:
        raise ValueError('Drop dimensions, duration and contact parameters must be positive and finite')
    if not np.isfinite(vertical_velocity) or vertical_velocity > 0:
        raise ValueError('Use a finite downward or zero initial vertical velocity')
    if integrator not in ('RK4', 'implicitfast', 'implicit', 'Euler'):
        raise ValueError('Unsupported integrator')
    xml = f'''<mujoco><option timestep="{timestep}" integrator="{integrator}"/>
      <worldbody><geom type="plane" size="5 5 .1" solref="{time_constant} {damping_ratio}"
      solimp=".95 .998 .002 .5 2"/>
      <body pos="0 0 {height}"><joint type="free" damping="{joint_damping}" armature="{armature}"/>
      <geom type="sphere" size="{radius}" mass="{mass}" condim="4" friction=".5 .05 .015"
      solref="{time_constant} {damping_ratio}" solimp=".95 .998 .002 .5 2"/></body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[2] = vertical_velocity
    hit = False
    peak = radius
    minimum = height
    for _ in range(int(duration/timestep)):
        mujoco.mj_step(model, data)
        minimum = min(minimum, float(data.qpos[2]))
        hit = hit or data.ncon > 0
        if hit:
            peak = max(peak, float(data.qpos[2]))
    # With zero spin and no active forces, the initial energy bounds the apex.
    energy_bound = height+vertical_velocity**2*(mass+armature)/(2*mass*9.81)
    return dict(height=height, hit_floor=bool(hit), max_height_after_first_floor=peak if hit else None,
                max_penetration=max(0., radius-minimum),
                effective_height_restitution=float(np.sqrt(max(0., peak-radius)/(energy_bound-radius))) if hit else None,
                passive_energy_height_bound=energy_bound,
                exceeds_energy_height_bound=bool(hit and peak > energy_bound+.005))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', nargs='+', type=float, default=[.002, .0005, .00025])
    parser.add_argument('--height', type=float, default=.9)
    parser.add_argument('--height-jitter', type=float, default=.002)
    parser.add_argument('--samples', type=int, default=17)
    parser.add_argument('--time-constant', type=float, default=.008)
    parser.add_argument('--damping-ratio', type=float, default=.1)
    parser.add_argument('--integrator', choices=['RK4', 'implicitfast', 'implicit', 'Euler'], default='RK4')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or not np.isfinite(args.height_jitter) or args.height_jitter < 0:
        parser.error('Use >=2 samples and finite nonnegative height jitter')
    from ycb_objects.loader import load, sphere_radius
    profiles = dict(default=(.05, .04))
    for name in ('tennis_ball', 'baseball'):
        resource = load(name)
        profiles[name] = (resource[1], sphere_radius(resource))
    report = dict(mujoco_version=mujoco.__version__, integrator=args.integrator, time_constant=args.time_constant,
                  damping_ratio=args.damping_ratio,
                  assumptions='No robot, zero horizontal speed/spin, centered homogeneous sphere; no claim of measured ball physics',
                  results=[])
    for name, (mass, radius) in profiles.items():
        for timestep in args.timesteps:
            rows = [drop(mass, radius, float(height), timestep,
                         time_constant=args.time_constant, damping_ratio=args.damping_ratio,
                         integrator=args.integrator)
                    for height in np.linspace(args.height-args.height_jitter,
                                              args.height+args.height_jitter, args.samples)]
            peaks = [r['max_height_after_first_floor'] for r in rows if r['hit_floor']]
            entry = dict(object=name, mass=mass, radius=radius, timestep=timestep, rows=rows,
                         min_peak=min(peaks) if peaks else None, max_peak=max(peaks) if peaks else None,
                         energy_bound_violations=sum(r['exceeds_energy_height_bound'] for r in rows))
            report['results'].append(entry)
            print(name, timestep, 'peak range', entry['min_peak'], entry['max_peak'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report, file, indent=2)


if __name__ == '__main__':
    main()
