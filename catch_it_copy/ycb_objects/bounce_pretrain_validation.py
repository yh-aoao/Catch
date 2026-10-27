"""Reproducible vertical-drop candidate/reference sweep, without robot contacts."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gym_dcmm.envs.bounce_stable import TIMESTEP, SOLREF
from ycb_objects.bounce_stability import drop


def validate(samples=60, seed=20261027):
    if samples < 1:
        raise ValueError('samples must be positive')
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(samples):
        mass, radius = float(rng.uniform(.04, .2)), float(rng.uniform(.025, .05))
        height, vz = float(rng.uniform(.6, 1.2)), float(rng.uniform(-1., 0.))
        a, b = [drop(mass, radius, height, dt, time_constant=SOLREF[0],
                     damping_ratio=SOLREF[1], vertical_velocity=vz)
                for dt in (TIMESTEP, TIMESTEP/2)]
        rows.append(dict(mass=mass, radius=radius, height=height, vz=vz,
                         candidate=a, reference=b,
                         peak_difference=abs(a['max_height_after_first_floor']-
                                             b['max_height_after_first_floor'])))
    result = dict(seed=seed, samples=samples, mujoco_version=mujoco.__version__,
                  timestep=TIMESTEP, reference_timestep=TIMESTEP/2, solref=SOLREF,
                  assumptions='Centered homogeneous sphere; no horizontal speed/spin/robot; not material calibration',
                  max_peak_difference=max(r['peak_difference'] for r in rows),
                  max_penetration=max(r['candidate']['max_penetration'] for r in rows),
                  energy_violations=sum(r['candidate']['exceeds_energy_height_bound'] for r in rows), rows=rows)
    result['passed'] = (result['max_peak_difference'] < .015 and
                        result['max_penetration'] < .005 and result['energy_violations'] == 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as file:
        json.dump(result, file, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
