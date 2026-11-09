"""Resolve matched configs and smoke-test all three targets and both stages."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'analysis_artifacts/three_ball_parameter_setup_20261109'
ENV=dict(os.environ,PATH='/home/isee/miniconda3/envs/dcmm/bin:'+os.environ['PATH'],
         OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',WANDB_MODE='disabled')

def run(job):
    motion,stage,variant=job
    args=['bash','scripts/train_three_ball_parameter_v1.sh',motion,stage,variant,'device_id=-1','hydra.job.chdir=false']
    if stage=='catch':
        model='roll_track_best_reward_25.16.pth' if motion=='roll' else 'bounce_track_best_reward_19.43.pth'
        args.append('checkpoint_tracking='+str(ROOT.parent/'11-6'/model))
    tag=f'{motion}_{stage}_{variant}'
    with (OUT/f'{tag}_resolved.yaml').open('w') as f:
        subprocess.run(args+['--cfg','job','--resolve'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.PIPE,check=True)
    with (OUT/f'{tag}_smoke.log').open('w') as f:
        p=subprocess.run(args+['num_envs=3','train.ppo.horizon_length=8','train.ppo.mini_epochs=1',
            'train.ppo.max_agent_steps=72','wandb_mode=disabled',f'output_name=three_ball_parameter_smoke_{tag}_20261109'],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    print(tag,p.returncode,flush=True)
    if p.returncode:print((OUT/f'{tag}_smoke.log').read_text()[-2000:],flush=True)
    return dict(motion=motion,stage=stage,variant=variant,exit_code=p.returncode)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(m,s,v) for m in ['roll','bounce'] for s in ['track','catch'] for v in ['dr','fixed']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(run,jobs))
    assert all(x['exit_code']==0 for x in results)
    for m in ['roll','bounce']:
        for s in ['track','catch']:
            a=OmegaConf.to_container(OmegaConf.load(OUT/f'{m}_{s}_dr_resolved.yaml'),resolve=True)
            b=OmegaConf.to_container(OmegaConf.load(OUT/f'{m}_{s}_fixed_resolved.yaml'),resolve=True)
            assert a['num_envs']==b['num_envs']==24
            assert a['benchmark_options']['domain_randomization']=='physical_dr_v1'
            assert b['benchmark_options']['domain_randomization']=='off'
            for d in (a,b):d.pop('output_name');d['benchmark_options'].pop('domain_randomization')
            assert a==b,'Unintended config difference'
    (OUT/'validation.json').write_text(json.dumps(results,indent=2))
    (OUT/'COMPLETE').write_text('8 PPO smoke runs passed; paired configs differ only in DR and output name\n')
