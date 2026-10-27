"""Passive Roll/table checks using exactly the physical_v1 proxy/contact builder."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gym_dcmm.envs.roll_physical import load_materials, table_pair


def trial(profile,speed=1.,timestep=.002,duration=4.,start=(0.,2.4),heading=0.,
          extended_table=False,rolling_length=None):
    p=dict(profile)
    if rolling_length is not None:
        p['rolling_length_m']=rolling_length
    if not np.isfinite([speed,timestep,duration,heading,*start]).all() or speed<0 or timestep<=0 or duration<=0:
        raise ValueError('Invalid trial conditions')
    size='10 10 .02' if extended_table else '1.2 1.6 .02'
    xml=f'''<mujoco><option timestep="{timestep}" integrator="RK4"/>
      <worldbody><geom name="table_surface" type="box" pos="0 2.5 .68" size="{size}"/>
      <body name="object"><joint type="free"/><geom name="object" type="sphere" size=".04"/></body>
      </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(table_pair(xml,p))
    data=mujoco.MjData(model)
    radius,mass=p['radius_m'],p['mass_kg']
    data.qpos[:3]=[*start,.7+radius+.0002]
    data.qpos[3:]=[1,0,0,0]
    vx,vy=speed*np.sin(heading),-speed*np.cos(heading)
    data.qvel[:]=[vx,vy,0,-vy/radius,vx/radius,0]
    edge=-7.5 if extended_table else .9
    def energy():
        return float(mass*9.81*data.qpos[2]+.5*mass*np.dot(data.qvel[:3],data.qvel[:3])+
                     .5*p['inertia_factor']*mass*radius**2*np.dot(data.qvel[3:],data.qvel[3:]))
    initial_energy=energy()
    max_energy=initial_energy
    samples=[]
    exit_state=None
    for index in range(int(duration/timestep)):
        mujoco.mj_step(model,data)
        max_energy=max(max_energy,energy())
        if index%max(1,round(.02/timestep))==0:
            samples.append(dict(time=float(data.time),position=data.qpos[:3].tolist(),
                                speed=float(np.linalg.norm(data.qvel[:2]))))
        if data.qpos[1]+radius<edge and not data.ncon:
            exit_state=dict(time=float(data.time),speed=float(np.linalg.norm(data.qvel[:2])),
                            position=data.qpos[:3].tolist())
            break
    return dict(profile=p,timestep=timestep,initial_speed=speed,heading=heading,
                exit=exit_state,final_speed=float(np.linalg.norm(data.qvel[:2])),
                max_passive_energy_increase_j=max_energy-initial_energy,
                finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),samples=samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    materials=load_materials()
    rng=np.random.RandomState(20261027)
    conditions=[(float(rng.uniform(.8,1.3)),(float(rng.uniform(-.25,.25)),float(rng.uniform(2.1,2.6))),
                 float(rng.uniform(-np.pi/6,np.pi/6))) for _ in range(10)]
    rows=[]
    for name,p in materials['profiles'].items():
        cases=[]
        for speed,start,heading in conditions:
            a=trial(p,speed=speed,start=start,heading=heading)
            b=trial(p,speed=speed,start=start,heading=heading,timestep=.001)
            cases.append(dict(candidate=a,reference=b))
        coast=trial(p,extended_table=True,duration=2.)
        no_rolling=trial(p,extended_table=True,duration=2.,rolling_length=0.)
        slow=trial(p,speed=.03,extended_table=True,duration=3.)
        rows.append(dict(object=name,cases=cases,coast=coast,no_rolling=no_rolling,slow=slow))
        print(name,'edge',sum(c['candidate']['exit'] is not None for c in cases),'/10',
              'coast v',coast['final_speed'],'no rolling v',no_rolling['final_speed'],
              'slow v',slow['final_speed'],flush=True)
    max_delta=max(abs(c['candidate']['exit']['speed']-c['reference']['exit']['speed'])
                  for r in rows for c in r['cases'] if c['candidate']['exit'] and c['reference']['exit'])
    report=dict(mujoco_version=mujoco.__version__,materials=materials,rows=rows,
                max_exit_speed_refinement_difference=max_delta,
                assumptions='Passive isolated ball/table. No claim of real material calibration or policy success.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as file:
        json.dump(report,file,indent=2)
    assert all(c['candidate']['finite'] and c['candidate']['exit'] for r in rows for c in r['cases'])
    assert max_delta<.03


if __name__=='__main__':
    main()
