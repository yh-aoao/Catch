"""Tiny CPU PPO updates: both Track entries, then new Bounce Catch from that Track."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/isee/Catch/catch_it_copy')
OUT = Path(__file__).resolve().parent
ENV = dict(os.environ, PATH='/home/isee/miniconda3/envs/dcmm/bin:'+os.environ.get('PATH',''),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
COMMON = ['device_id=-1','num_envs=2','train.ppo.num_actors=2','vector_mode=sync',
          'train.ppo.horizon_length=8','train.ppo.minibatch_size=16','train.ppo.mini_epochs=1',
          'train.ppo.max_agent_steps=64','train.ppo.save_best_after=0','train.ppo.save_frequency=1']


def run(motion, task='Tracking', checkpoint=None):
    label = motion+('_track' if task=='Tracking' else '_catch')
    name = 'equal_posture_smoke_'+label+'_20261101'
    command = ['bash', 'scripts/train_'+motion+'_three_ball_equal_v2.sh',
               'task='+task, *COMMON, 'output_name='+name]
    if checkpoint:
        command.append('checkpoint_tracking='+str(checkpoint))
    with (OUT/(label+'_training_smoke.log')).open('w') as file:
        result = subprocess.run(command,cwd=ROOT,env=ENV,stdout=file,stderr=subprocess.STDOUT)
    assert result.returncode == 0, label+' failed; inspect log'
    prefix = motion+('_track' if task=='Tracking' else '_catch')
    checkpoints = sorted((ROOT/'outputs'/name).glob('**/nn/'+prefix+'_last.pth'),
                         key=lambda path:path.stat().st_mtime)
    assert checkpoints, label+' did not save last'
    print(label,checkpoints[-1],flush=True)
    return dict(label=label,command=command,checkpoint=str(checkpoints[-1]),returncode=0)


with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(run, ['roll','bounce']))
track = next(r['checkpoint'] for r in results if r['label']=='bounce_track')
results.append(run('bounce','Catching_TwoStage',track))
(OUT/'training_smoke.json').write_text(json.dumps(results,indent=2))
print('Three short training runs completed; no formal policy quality claim.',flush=True)
