"""Passive drops of all six baseline balls at DR damping/size/mass corners."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gym_dcmm.envs.object_benchmark import catalog
from gym_dcmm.envs.bounce_physical_v2 import profiles
from ycb_objects.benchmark_geometry import geometry

def drop(mass,radius,time_constant,damping,height,dt):
    xml=f'''<mujoco><option timestep="{dt}" integrator="RK4"/><worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body pos="0 0 {height+radius}"><freejoint/><geom name="ball" type="sphere" size="{radius}" mass="{mass}"/></body>
    </worldbody><contact><pair geom1="floor" geom2="ball" condim="4" friction=".3 .3 .0001 0 0"
    solref="{time_constant} {damping}" solimp=".95 .998 .002 .5 2"/></contact></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    hit=False;up=False;peak=0.;penetration=0.
    for _ in range(int(3/dt)):
        mujoco.mj_step(m,d)
        if d.ncon:hit=True
        penetration=max(penetration,max(0.,radius-d.qpos[2]))
        if hit and d.qvel[2]>0:up=True;peak=max(peak,float(d.qpos[2]-radius))
        if up and d.qvel[2]<=0:break
    if not up:raise AssertionError('No bounce')
    return dict(apex=peak,cor=float(np.sqrt(peak/height)),penetration=penetration)

def main():
    cat=catalog(ROOT/'ycb_objects/calibration_data/object_benchmark_all_targets_v1.json')
    old=json.loads((ROOT/'ycb_objects/calibration_data/balls.three_ball_training_v1.json').read_text())['profiles']
    rows=[]
    for name,p in cat['bounce'].items():
        if name in old:
            mass,radius,_=profiles()[name];p=old[name]
        else:
            g=geometry(name,'sphere',p['inertia_factor']);mass,radius=g['mass'],g['radius']
        for height in (.5,1.5):
            base=drop(mass,radius,p['time_constant'],p['damping_ratio'],height,.000125)
            for scale in (.9,1.,1.1):
                # Include opposing mass and radius extremes.
                ms=1.1 if scale<1 else .9;rs=.95 if scale<1 else 1.05
                a=drop(mass*ms,radius*rs,p['time_constant'],p['damping_ratio']*scale,height,.000125)
                b=drop(mass*ms,radius*rs,p['time_constant'],p['damping_ratio']*scale,height,.0000625)
                assert 0<a['cor']<1 and abs(a['cor']-base['cor'])<.06,(name,a,base)
                assert abs(a['apex']-b['apex'])<.015,(name,a,b)
                assert a['penetration']<.02,(name,a)
                rows.append(dict(object=name,height=height,damping_scale=scale,baseline=base,coarse=a,fine=b))
    out=ROOT.parent/'analysis_artifacts/physical_dr_v1_20261105/drop_validation.json'
    out.write_text(json.dumps(rows,indent=2));print('PASS',len(rows),'drop comparisons')
if __name__=='__main__':main()
