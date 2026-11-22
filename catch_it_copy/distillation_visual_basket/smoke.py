"""Small real-renderer/parallel-collector/backprop check; not a success-rate evaluation."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import copy,json
import numpy as np
import torch
from .run import Collector,optimize
from .core import Student,Replay,TASKS
from .teachers import FrozenTeacher
from distillation.core import ActionAdapter


def main():
    torch.set_num_threads(1)
    c=json.loads((Path(__file__).parent/'configs/catch.json').read_text())
    c.update(device='cpu',num_envs_per_task=1,bptt_steps=2,batch_episodes_per_task=1)
    for t in TASKS:c['tasks'][t]['objects']=[c['tasks'][t]['objects'][0]]
    c['tasks']['roll']['objects']=['orange'];c['tasks']['bounce']['objects']=['golf_ball']
    c['vision']['gl_backend']='egl'
    teachers={t:FrozenTeacher(t,v,'cpu') for t,v in c['tasks'].items()}
    student=Student(c);replay=Replay(c);adapters={t:ActionAdapter(v) for t,v in c['tasks'].items()}
    collector=Collector(c,student,teachers,adapters,replay)
    try:
        collector.collect(2)
        student.fit_normalizer(np.concatenate(collector.states));collector.calibrating=False
        for task in TASKS:
            # Deliberate short truncation for shape/backprop validation only.
            replay.add(task,c['tasks'][task]['objects'][0],collector.pending[task][0])
        batch=replay.sample(1,'cpu');optimizer=torch.optim.Adam(student.parameters(),lr=1e-4)
        metrics=optimize(student,optimizer,batch,c)
        assert all(np.isfinite(v) for v in metrics.values())
        assert student.basket_position_head[-1].weight.grad is not None
        collector.refresh_hidden()
        print(json.dumps(dict(parallel_workers=4,steps_per_worker=2,loss=metrics['loss'],basket_position_loss=metrics['basket_position_loss'],
            basket_center_in_camera_frustum=collector.pools['basket'].obs['basket_center_in_camera_frustum'].tolist(),
            max_context_tail=max(float(np.abs(p.obs['context'][:,1:]).max()) for p in collector.pools.values())),indent=2))
    finally:collector.close()

if __name__=='__main__':main()
