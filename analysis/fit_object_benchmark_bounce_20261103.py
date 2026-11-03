import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"catch_it_copy"))
from scipy.optimize import minimize_scalar
from ycb_objects.benchmark_geometry import geometry
from ycb_objects.bounce_stability import drop
p=ROOT/'catch_it_copy/ycb_objects/calibration_data/object_benchmark_v1.json';d=json.loads(p.read_text());report={}
for name,profile in d['bounce'].items():
 s=geometry(name,'sphere',profile['inertia_factor']);m,r=s['mass'],s['radius'];target=profile['normal_cor_target']
 def predict(ratio,h=1.,dt=.000125):return drop(m,r,h+r,dt,duration=1.5,time_constant=profile["time_constant"],damping_ratio=ratio,armature=0.,joint_damping=0.,vertical_velocity=0.)
 opt=minimize_scalar(lambda z:(predict(z)['effective_height_restitution']-target)**2,bounds=(.04,.65),method='bounded',options={'xatol':1e-5})
 ratio=float(opt.x);profile['damping_ratio']=ratio
 coarse=[predict(ratio,h) for h in (.5,1.,1.5)];fine=[predict(ratio,h,.0000625) for h in (.5,1.,1.5)]
 error=max(abs(a['max_height_after_first_floor']-b['max_height_after_first_floor']) for a,b in zip(coarse,fine))
 passed=bool(error<.015 and max(abs(x['effective_height_restitution']-target) for x in coarse)<.025 and not any(x['exceeds_energy_height_bound'] for x in coarse+fine))
 result={'mass_kg':m,'radius_m':r,'target_cor':target,'damping_ratio':ratio,'coarse':coarse,'fine':fine,'refinement_error_m':error,'passed':passed}
 report[name]=result;profile['numerical_validation']={'passed':passed,'peak_refinement_error_m':error,'calibration':'simulation-only fit to nominal COR; not real measurements'}
 print(name,ratio,passed,flush=True)
 assert passed
p.write_text(json.dumps(d,indent=2)+'\n')
(ROOT/'analysis_artifacts/object_benchmark_20261103/drop_validation.json').write_text(json.dumps(report,indent=2))
