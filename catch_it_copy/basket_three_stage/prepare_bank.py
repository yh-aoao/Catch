"""Validate recorded Catch poses in Basket tennis physics; never invent a missing source."""
import argparse,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();os.chdir(ROOT)
 from basket_autonomous.environment import AutonomousTeacherEnv
 from basket_three_stage.environment import AutonomousTeacherEnv as ThreeEnv
 from types import MethodType
 records=[r for file in a.inputs for r in json.loads(file.read_text())]
 c=json.loads((ROOT/'basket_autonomous/configs/fixed_carry_v2.json').read_text());opt=c['options'];opt.pop('pose_bank',None);opt['pose_bank_entries']=records;opt['reset_attempts']=4
 env=AutonomousTeacherEnv(opt);accepted=[];rejected=[]
 try:
  env._accept_hold_placement=MethodType(ThreeEnv._accept_hold_placement,env)
  env._accept_prepared_hold=MethodType(ThreeEnv._accept_prepared_hold,env)
  env._hold_noise_scales=lambda anchor:(np.zeros(6),0.,np.zeros(3))
  for i,record in enumerate(records):
   env._sampling_condition=('recorded',record['task']);env._sample_hold_anchor=lambda r=record:r
   try:
    np.random.seed(91200+i);_,info=env.reset(seed=91200+i)
    entry=dict(record,basket_validation=dict(placement=info['reset_sampling']['placement'],
      local=info['reset_sampling']['ball_local'],arm=info['reset_sampling']['arm'],hand=info['reset_sampling']['hand']))
    accepted.append(entry)
   except RuntimeError as error:rejected.append(dict(source=record['source'],task=record['task'],reason=str(error)))
   print('validate',i+1,'/',len(records),'accepted',len(accepted),flush=True)
 finally:env.close()
 counts={task:sum(r['task']==task for r in accepted) for task in ('throw','roll','bounce')}
 report=dict(counts=counts,total_accepted=len(accepted),rejected=rejected,inputs=[str(p.resolve()) for p in a.inputs])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
 if min(counts.values())<4:raise RuntimeError('Insufficient validated coverage: '+str(counts))
 a.output.write_text(json.dumps(accepted,indent=2));print('Validated bank:',counts,flush=True)
if __name__=='__main__':main()
