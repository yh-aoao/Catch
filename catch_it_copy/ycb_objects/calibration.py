"""Collect real-ball measurements and extract empirical motion targets.

This tool never modifies training physics or claims that a measured trajectory
uniquely identifies MuJoCo friction, damping, inertia or restitution parameters.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import json
from numbers import Real
from pathlib import Path
import sys

import numpy as np

SCHEMA = 'ycb_real_measurements_v1'


def number(value, name, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    value = float(value)
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(name + ' is outside its allowed range')
    return value


def template(name):
    return dict(schema=SCHEMA, object=name, specimen_id=None, mass_g=None,
                diameters_mm=[None, None, None],
                surfaces=dict(table=None, floor=None),
                rolling_trials=[dict(trial_id='roll_01', surface_id='table', fps=None,
                                     moving_only=True,
                                     samples=[dict(t_s=None, distance_m=None) for _ in range(5)])],
                bounce_trials=[dict(trial_id='bounce_01', surface_id='floor', fps=None,
                                    release_clearance_m=None, rebound_clearance_m=None)],
                notes='Use the same physical ball and surfaces as the robot experiment. Heights are ball-bottom clearance above the surface.')


def fit_rolling(trial):
    if trial.get('moving_only') is not True:
        raise ValueError('Rolling fit requires moving_only=true; exclude stationary samples')
    samples = trial['samples']
    if len(samples) < 4:
        raise ValueError('At least four moving samples are required')
    t = np.array([number(v['t_s'], 't_s', nonnegative=True) for v in samples])
    x = np.array([number(v['distance_m'], 'distance_m') for v in samples])
    if np.any(np.diff(t) <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError('Time and forward distance must increase; split reversals or stopped sections into separate trials')
    t -= t[0]
    design = np.stack([np.ones_like(t), t, .5*t*t], axis=1)
    fitted, _, rank, _ = np.linalg.lstsq(design, x, rcond=None)
    if rank < 3:
        raise ValueError('Rolling samples do not constrain a quadratic trajectory')
    offset, speed, acceleration = map(float, fitted)
    deceleration = max(0., -acceleration)
    return dict(trial_id=trial['trial_id'], surface_id=trial['surface_id'],
                sample_count=len(samples), duration_s=float(t[-1]),
                initial_speed_m_s=speed, acceleration_m_s2=acceleration,
                final_speed_fit_m_s=float(speed+acceleration*t[-1]),
                position_rmse_m=float(np.sqrt(np.mean((design@fitted-x)**2))),
                model='constant acceleration empirical summary, not a contact coefficient',
                stopping_distance_extrapolation_m=(speed**2/(2*deceleration) if deceleration > 1e-8 else None),
                stopping_time_extrapolation_s=(speed/deceleration if deceleration > 1e-8 else None),
                warnings=(['No measured deceleration; inspect slope, release or recording'] if acceleration >= 0 else []))


def analyze(data):
    if data.get('schema') != SCHEMA:
        raise ValueError('Unexpected measurement schema')
    if data.get('object') not in ('tennis_ball', 'baseball'):
        raise ValueError('Only tennis_ball and baseball are supported')
    mass = data.get('mass_g')
    if mass is not None:
        mass = number(mass, 'mass_g', positive=True)/1000
    diameters = data.get('diameters_mm', [None]*3)
    if len(diameters) != 3:
        raise ValueError('Record three perpendicular diameters')
    measured_diameters = [number(v, 'diameter_mm', positive=True) for v in diameters if v is not None]
    radius = float(np.mean(measured_diameters)/2000) if len(measured_diameters) == 3 else None
    result = dict(object=data['object'], specimen_id=data.get('specimen_id'),
                  mass_kg=mass, sphere_radius_m=radius, rolling=[], bounce=[], incomplete_trials=[],
                  surfaces=data.get('surfaces', {}), mujoco_parameters_identified=False,
                  inertia_identified=False, warnings=[])
    if radius and np.ptp(measured_diameters)/np.mean(measured_diameters) > .03:
        result['warnings'].append('Diameter spread exceeds 3%; recheck measurement and spherical-shape approximation')
    for trial in data.get('rolling_trials', []):
        if not trial.get('samples') or any(v.get('t_s') is None or v.get('distance_m') is None for v in trial['samples']):
            result['incomplete_trials'].append(trial['trial_id'])
            continue
        result['rolling'].append(fit_rolling(trial))
    for trial in data.get('bounce_trials', []):
        release, rebound = trial.get('release_clearance_m'), trial.get('rebound_clearance_m')
        if release is None or rebound is None:
            result['incomplete_trials'].append(trial['trial_id'])
            continue
        release = number(release, 'release_clearance_m', positive=True)
        rebound = number(rebound, 'rebound_clearance_m', nonnegative=True)
        if rebound > release:
            raise ValueError('Rebound height exceeds release height; check release impulse and height definitions')
        result['bounce'].append(dict(trial_id=trial['trial_id'], surface_id=trial['surface_id'],
                                      release_clearance_m=release, rebound_clearance_m=rebound,
                                      effective_restitution_height_ratio=float(np.sqrt(rebound/release)),
                                      note='Zero-velocity release; ignores air drag. This is not solref damping ratio.'))
    result['status'] = ('empirical_targets_available' if result['rolling'] or result['bounce'] else
                        'basic_dimensions_only' if mass is not None or radius is not None else
                        'awaiting_measurements')
    return result


def inspect_current():
    with redirect_stdout(sys.stderr):
        from gym_dcmm.envs.object_test import make_object_test
    result = dict(status='simulation_reference_not_real_measurements', objects={})
    for name in ('tennis_ball', 'baseball'):
        result['objects'][name] = {}
        for motion in ('roll', 'bounce'):
            with redirect_stdout(sys.stderr):
                env = make_object_test('ycb:'+name, task='Catching', object_motion=motion,
                                       object_eval=False, viewer=False, render_mode=None)
                try:
                    env.reset()
                    model = env.Dcmm.model
                    bid = model.body('object').id
                    dof = model.jnt_dofadr[model.body_jntadr[bid]]
                    gid = int(np.flatnonzero((model.geom_bodyid == bid) & (model.geom_contype != 0))[0])
                    surface = model.geom('table_surface' if motion == 'roll' else 'floor').id
                    result['objects'][name][motion] = dict(
                        mass_kg=float(model.body_mass[bid]), sphere_radius_m=float(model.geom_size[gid, 0]),
                        principal_inertia_kg_m2=model.body_inertia[bid].tolist(),
                        center_of_mass_m=model.body_ipos[bid].tolist(),
                        translational_armature_kg=model.dof_armature[dof:dof+3].tolist(),
                        rotational_armature_kg_m2=model.dof_armature[dof+3:dof+6].tolist(),
                        free_joint_damping=model.dof_damping[dof:dof+6].tolist(),
                        ball_contact=dict(condim=int(model.geom_condim[gid]),
                                          friction=model.geom_friction[gid].tolist(),
                                          solref=model.geom_solref[gid].tolist(),
                                          solimp=model.geom_solimp[gid].tolist(),
                                          priority=int(model.geom_priority[gid])),
                        surface_contact=dict(friction=model.geom_friction[surface].tolist(),
                                             solref=model.geom_solref[surface].tolist(),
                                             solimp=model.geom_solimp[surface].tolist(),
                                             condim=int(model.geom_condim[surface]),
                                             priority=int(model.geom_priority[surface])))
                finally:
                    env.close()
    return result


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['template', 'inspect', 'analyze'])
    parser.add_argument('--object', choices=['tennis_ball', 'baseball'], default='tennis_ball')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'template':
        result = template(args.object)
    elif args.command == 'inspect':
        result = inspect_current()
        result['recorded_local_time'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    else:
        if args.input is None:
            parser.error('analyze requires --input')
        content = args.input.read_bytes()
        result = analyze(json.loads(content))
        result['input_sha256'] = hashlib.sha256(content).hexdigest()
    write_new(args.output, result)
    print('Written:', args.output)


if __name__ == '__main__':
    main()
