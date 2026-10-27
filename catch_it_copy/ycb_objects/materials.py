"""Per-specimen, per-surface Bounce materials; no invented real measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'ball_ground_materials_v1'
NAMES = ('original', 'tennis_ball', 'baseball')


def finite(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be finite in [{low}, {high}]')
    return float(value)


def load(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('schema') != SCHEMA or not isinstance(data.get('surface_id'), str) or not data['surface_id'].strip():
        raise ValueError('A ball_ground_materials_v1 file requires a nonempty surface_id')
    profiles = data.get('profiles', {})
    if not profiles or set(profiles)-set(NAMES):
        raise ValueError('Use profiles original/tennis_ball/baseball')
    for name, p in profiles.items():
        if p.get('bounce_status') not in ('simulation_prior', 'fitted_drop'):
            raise ValueError(name+': bounce_status must be simulation_prior or fitted_drop')
        if p.get('friction_status') not in ('simulation_prior', 'fitted_trajectory'):
            raise ValueError(name+': friction_status must be simulation_prior or fitted_trajectory')
        for key, limits in dict(damping_ratio=(.04, 1.), time_constant=(.008, .05),
                                sliding_friction=(0., 3.), torsional_friction=(0., .1)).items():
            finite(p.get(key), name+'.'+key, *limits)
        for key, limits in dict(mass_kg=(.001, 1.), radius_m=(.005, .15)).items():
            if p.get(key) is not None:
                finite(p[key], name+'.'+key, *limits)
        if p.get('inertia_factor') is not None:
            finite(p['inertia_factor'], name+'.inertia_factor', .1, 2/3)
        if p['bounce_status'] == 'fitted_drop':
            if not p.get('measurement_sha256') or not p.get('specimen_id'):
                raise ValueError(name+': fitted_drop requires specimen_id and measurement_sha256')
            validation = p.get('validation', {})
            if validation.get('passed') is not True:
                raise ValueError(name+': fitted drop must pass refinement and fit checks')
            finite(validation.get('peak_refinement_error_m'), name+'.refinement', 0., .015)
            finite(validation.get('fit_rmse_m'), name+'.fit_rmse', 0., .02)
    return dict(data, path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def fit_drop(measurements, time_constant=.008):
    """Fit one normal damping parameter; friction/compliance are not identified."""
    from scipy.optimize import minimize_scalar
    from ycb_objects.calibration import analyze
    from ycb_objects.bounce_stability import drop
    from gym_dcmm.envs.bounce_physical_v2 import profiles, DT
    finite(time_constant, 'time_constant', .008, .05)
    targets = analyze(measurements)
    specimen = measurements.get('specimen_id')
    surface = measurements.get('surfaces', {}).get('floor')
    if not isinstance(specimen, str) or not specimen.strip() or not isinstance(surface, str) or not surface.strip():
        raise ValueError('Record specimen_id and the actual floor description first')
    trials = targets['bounce']
    if len(trials) < 3 or any(t['surface_id'] != 'floor' for t in trials):
        raise ValueError('Provide >=3 complete zero-velocity floor drop trials; table trials must be separate')
    heights = [t['release_clearance_m'] for t in trials]
    if max(heights)-min(heights) < .1:
        raise ValueError('Use at least two release clearances separated by >=0.1 m')
    mass0, radius0, _ = profiles()[targets['object']]
    mass, radius = targets['mass_kg'] or mass0, targets['sphere_radius_m'] or radius0
    def predictions(ratio, dt=DT):
        return [drop(mass, radius, h+radius, dt,
                     duration=max(2., 3*np.sqrt(2*h/9.81)), time_constant=time_constant,
                     damping_ratio=ratio, armature=0., joint_damping=0., vertical_velocity=0.)
                for h in heights]
    observed = np.array([t['rebound_clearance_m'] for t in trials])
    def loss(ratio):
        return float(np.mean((np.array([r['max_height_after_first_floor']-radius
                                        for r in predictions(ratio)])-observed)**2))
    result = minimize_scalar(loss, method='bounded', bounds=(.04, 1.), options=dict(xatol=1e-4))
    ratio = float(result.x)
    coarse, fine = predictions(ratio), predictions(ratio, DT/2)
    error = max(abs(a['max_height_after_first_floor']-b['max_height_after_first_floor']) for a,b in zip(coarse,fine))
    rmse = float(np.sqrt(loss(ratio)))
    # These are identification checks on the input data, not held-out validation.
    passed = bool(result.success and error <= .015 and rmse <= .02 and
                  not any(r['exceeds_energy_height_bound'] for r in coarse))
    return dict(specimen_id=measurements['specimen_id'], bounce_status='fitted_drop',
                friction_status='simulation_prior', mass_kg=targets['mass_kg'], radius_m=targets['sphere_radius_m'],
                damping_ratio=ratio, time_constant=time_constant, sliding_friction=.3, torsional_friction=.005,
                measurement_sha256=hashlib.sha256(json.dumps(measurements,sort_keys=True).encode()).hexdigest(),
                validation=dict(passed=passed, training_trial_count=len(trials), fit_rmse_m=rmse,
                                peak_refinement_error_m=float(error), held_out_validated=False),
                limitations='Normal bounce fit only; friction .3/.005 and compliance are priors; inertia is structural approximation; validate separate drops/oblique impacts before real transfer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fit-drop', 'inspect', 'merge'])
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--with-file', type=Path, action='append', default=[])
    parser.add_argument('--time-constant', type=float, default=.008)
    args = parser.parse_args()
    if args.command == 'inspect':
        print(json.dumps(load(args.input), indent=2, ensure_ascii=False))
        return
    if args.output is None:
        parser.error('fit-drop/merge requires --output (new file)')
    if args.command == 'merge':
        sources = [load(p) for p in [args.input]+args.with_file]
        if len({s['surface_id'] for s in sources}) != 1:
            parser.error('Do not merge profiles measured on different surfaces')
        profiles = {}
        for source in sources:
            if set(profiles) & set(source['profiles']):
                parser.error('Duplicate object profile')
            profiles.update(source['profiles'])
        result = dict(schema=SCHEMA, surface_id=sources[0]['surface_id'], profiles=profiles,
                      input_sha256={s['path']:s['sha256'] for s in sources})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        print(args.output)
        return
    data = json.loads(args.input.read_text())
    try:
        profile = fit_drop(data, time_constant=args.time_constant)
    except ValueError as exc:
        parser.error(str(exc))
    report = dict(schema=SCHEMA, surface_id=data['surfaces']['floor'], profiles={data['object']:profile})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report,file,indent=2,ensure_ascii=False)
    print(args.output, profile['validation'])
    if not profile['validation']['passed']:
        raise SystemExit('Fit/refinement failed; report saved for diagnosis, not runtime loading')


if __name__ == '__main__':
    main()
