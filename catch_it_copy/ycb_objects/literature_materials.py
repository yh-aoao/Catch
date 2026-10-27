"""Reproducible literature-informed priors, never represented as specimen data."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from ycb_objects.bounce_stability import drop
from ycb_objects.materials import SCHEMA

SOURCES = {
    'ycb': 'https://arxiv.org/pdf/1502.03143',
    'cross_1999': 'https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/BallBounce.pdf',
    'cross_2002': 'https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/GripSlip.pdf',
    'baseball_overview': 'https://physics.usyd.edu.au/~cross/baseball.html',
}


def build():
    from gym_dcmm.envs.bounce_physical_v2 import DT, profiles
    result = dict(schema=SCHEMA, surface_id='generic_dry_rigid_floor_literature_approx_v1',
                  basis='literature_informed_simulation_prior_not_YCB_official_or_specimen_measurements',
                  sources=SOURCES, profiles={})
    # Tennis friction: court values >=.5 reported by Cross; generic floor transfer
    # is an engineering approximation. Baseball friction has no same-surface
    # measurement in these sources; .3 is explicitly a chosen sensitivity prior.
    choices = dict(tennis_ball=(.75,.5,.55), baseball=(.5,.3,.4))
    heights = (.5, 1., 1.5)
    for name,(target,mu,factor) in choices.items():
        mass,radius,_=profiles()[name]
        def simulate(ratio,dt=DT):
            return [drop(mass,radius,h+radius,dt,duration=2.,time_constant=.008,
                         damping_ratio=ratio,armature=0.,joint_damping=0.,vertical_velocity=0.)
                    for h in heights]
        def loss(ratio):
            return float(np.mean([(p['effective_height_restitution']-target)**2
                                  for p in simulate(ratio)]))
        fit=minimize_scalar(loss,method='bounded',bounds=(.04,1.),options=dict(xatol=1e-4))
        coarse,fine=simulate(float(fit.x)),simulate(float(fit.x),DT/2)
        error=max(abs(a['max_height_after_first_floor']-b['max_height_after_first_floor'])
                  for a,b in zip(coarse,fine))
        cor_error=max(abs(p['effective_height_restitution']-target) for p in coarse)
        passed=bool(fit.success and error<.015 and cor_error<.02 and
                    not any(p['exceeds_energy_height_bound'] for p in coarse))
        result['profiles'][name]=dict(bounce_status='simulation_prior',friction_status='simulation_prior',
            basis='literature_informed_approximation',normal_cor_target=target,
            damping_ratio=float(fit.x),time_constant=.008,sliding_friction=mu,
            torsional_friction=.005,inertia_factor=factor,
            friction_basis=('Cross 2002 reports tennis/court sliding friction usually >=0.5; transfer to generic floor is approximate'
                            if name=='tennis_ball' else 'Engineering prior .3; not a directly measured baseball/floor coefficient'),
            inertia_basis='Cross 2002 spherical-structure approximation; not measured YCB inertia',
            numerical_validation=dict(passed=passed,heights_clearance_m=list(heights),
                peak_refinement_error_m=float(error),max_cor_target_error=float(cor_error),
                coarse=coarse,fine=fine),
            limitations='Surface/speed/specimen dependent; fixed COR and Coulomb friction approximate compliant balls; torsional friction remains a prior')
        if not passed:
            raise ValueError(name+': numerical check failed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=build()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2)
    for name,p in result['profiles'].items():
        print(name,'COR target',p['normal_cor_target'],'damping_ratio',p['damping_ratio'],
              'mu',p['sliding_friction'],'validated',p['numerical_validation']['passed'])


if __name__ == '__main__':
    main()
