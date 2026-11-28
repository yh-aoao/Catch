"""Sequential single/multi-GPU rendering benchmark with identical learning work."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time

FIELDS=('seconds','collect_seconds','student_inference_seconds','teacher_inference_seconds',
        'environment_wait_seconds','collect_other_seconds','update_seconds','refresh_hidden_seconds')


def summarize(rows):
    if not rows:raise ValueError('No measured iterations')
    result={'iterations':len(rows)}
    for key in FIELDS:
        values=[r[key] for r in rows]
        result[key]={'mean':statistics.mean(values),'median':statistics.median(values),
                     'min':min(values),'max':max(values)}
    result['policy_steps_per_second']=sum(r['collected_policy_steps'] for r in rows)/sum(r['seconds'] for r in rows)
    return result


def worker(args):
    # Each placement starts in a fresh process, including CUDA, RNG and all workers.
    import numpy as np
    import torch
    from .run import (validate,seed_all,Student,Replay,FrozenTeacher,ActionAdapter,
                      Collector,TASKS,optimize,learning_rate)
    config=json.loads(args.config.read_text())
    config=copy.deepcopy(config);config['device']=args.device;config['vision']['gl_backend']='egl'
    config.pop('render_task_devices',None);config['render_devices']=args.render_devices
    if args.num_envs is not None:config['num_envs_per_task']=args.num_envs
    if args.rollout_steps is not None:config['rollout_steps']=args.rollout_steps
    validate(config)
    from .runtime import validate_torch_device
    args.device=str(validate_torch_device(args.device));config['device']=args.device
    torch.set_num_threads(1);seed_all(config['seed'])
    student=Student(config).to(args.device)
    if args.checkpoint:
        from .core import SCHEMA
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if saved.get('schema')!=SCHEMA:raise ValueError('Expected visual Bridge student checkpoint')
        # Environment, teacher, and learning settings must match the tested run.
        for key in ('tasks','env_kwargs','vision','model','auxiliary','sampling_mode',
                    'ablation','basket_rollin','basket_carry_precision_weight','capacity_frames_per_task','batch_episodes_per_object','updates_per_iteration','bptt_steps'):
            expected=copy.deepcopy(saved['config'].get(key));actual=copy.deepcopy(config.get(key))
            if key=='vision':
                expected.pop('gl_backend',None);actual.pop('gl_backend',None)
            if expected!=actual:raise ValueError('Checkpoint/config mismatch: '+key)
        student.load_state_dict(saved['student'],strict=True)
        start=saved['iteration']
    else:start=0
    teachers={t:FrozenTeacher(t,c,args.device) for t,c in config['tasks'].items()}
    adapters={t:ActionAdapter(c) for t,c in config['tasks'].items()}
    replay=Replay(config);optimizer=torch.optim.Adam(student.parameters(),lr=config['learning_rate'])
    def sync():
        if student.mean.device.type=='cuda':torch.cuda.synchronize(student.mean.device)
    collector=Collector(config,student,teachers,adapters,replay,start)
    rows=[]
    try:
        if collector.calibrating:
            collector.collect(config['calibration_steps']);student.fit_normalizer(np.concatenate(collector.states))
            collector.states.clear();collector.calibrating=False;collector.refresh_hidden()
        for _ in range(config['max_episode_steps']+1):
            if replay.ready():break
            collector.collect(1)
        if not replay.ready():raise RuntimeError('Replay warmup incomplete')
        (args.output/'resolved_config.json').write_text(json.dumps(config,indent=2)+'\n')
        for index in range(1,args.warmup+args.iterations+1):
            iteration=start+index
            for group in optimizer.param_groups:group['lr']=learning_rate(config,iteration)
            collector.teacher_fraction=max(config['teacher_mix_min'],config['teacher_mix_start']*(1-iteration/config['teacher_mix_decay']))
            sync();before=time.monotonic();row=collector.collect(config['rollout_steps'])
            stamp=time.monotonic()
            for _ in range(config['updates_per_iteration']):
                optimize(student,optimizer,replay.sample(config['batch_episodes_per_object'],'cpu'),config)
            sync();row['update_seconds']=time.monotonic()-stamp
            stamp=time.monotonic();collector.refresh_hidden();sync()
            row['refresh_hidden_seconds']=time.monotonic()-stamp;row['seconds']=time.monotonic()-before
            row.update(iteration=iteration,warmup=index<=args.warmup,
                       collected_policy_steps=4*config['num_envs_per_task']*config['rollout_steps'])
            with (args.output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print('[benchmark] {}/{} warmup={} total={:.2f}s collect={:.2f}s update={:.2f}s'.format(
                index,args.warmup+args.iterations,row['warmup'],row['seconds'],row['collect_seconds'],row['update_seconds']),flush=True)
            if not row['warmup']:rows.append(row)
        (args.output/'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')
    finally:collector.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--single-render-device',type=int)
    parser.add_argument('--multi-render-devices',type=int,nargs='+')
    parser.add_argument('--iterations',type=int,default=50);parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--num-envs',type=int);parser.add_argument('--rollout-steps',type=int)
    parser.add_argument('--checkpoint',type=Path,help='Optional same student snapshot for both tests; does not resume production training')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--render-devices',type=int,nargs='+',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.iterations<1 or args.warmup<0 or args.repeats<1:parser.error('Invalid iteration/warmup/repeat counts')
    args.config=args.config.expanduser().resolve();args.output=args.output.expanduser().resolve()
    if args.checkpoint:args.checkpoint=args.checkpoint.expanduser().resolve()
    if args.output.exists():parser.error('Choose a new output directory')
    if not args.worker:
        if args.single_render_device is None or args.single_render_device<0:parser.error('--single-render-device is required and nonnegative')
        if not args.multi_render_devices or len(set(args.multi_render_devices))<2 or any(x<0 for x in args.multi_render_devices):
            parser.error('--multi-render-devices requires at least two distinct nonnegative EGL IDs')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/catch_mpl')
    args.output.mkdir(parents=True)
    if args.worker:return worker(args)
    frozen=args.output/'input_config.json'
    frozen.write_text(args.config.read_text());args.config=frozen
    if args.checkpoint:
        frozen=args.output/'initial_student.pth';shutil.copyfile(args.checkpoint,frozen);args.checkpoint=frozen
    results={'single':[],'multi':[]}
    for repeat in range(args.repeats):
        # Alternate ordering across repeats to reduce warm-cache/order bias.
        order=('single','multi') if repeat%2==0 else ('multi','single')
        for mode in order:
            output=args.output/('{}_{:02d}'.format(mode,repeat+1))
            devices=[args.single_render_device] if mode=='single' else args.multi_render_devices
            command=[sys.executable,'-m','distillation_visual_bridge.benchmark','--worker',
                     '--config',str(args.config),'--output',str(output),'--device',args.device,
                     '--iterations',str(args.iterations),'--warmup',str(args.warmup),'--render-devices',*map(str,devices)]
            for flag,value in (('--checkpoint',args.checkpoint),('--num-envs',args.num_envs),('--rollout-steps',args.rollout_steps)):
                if value is not None:command.extend([flag,str(value)])
            print('[benchmark] starting {} repeat={} EGL={}'.format(mode,repeat+1,devices),flush=True)
            with (args.output/('{}_{:02d}.log'.format(mode,repeat+1))).open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('{} failed; see {}'.format(mode,log.name))
            summary=json.loads((output/'summary.json').read_text());results[mode].append(summary)
            print('[benchmark] {} mean={:.2f}s/iteration'.format(mode,summary['seconds']['mean']),flush=True)
    single=statistics.mean(r['seconds']['mean'] for r in results['single'])
    multi=statistics.mean(r['seconds']['mean'] for r in results['multi'])
    report=dict(results=results,single_seconds=single,multi_seconds=multi,speedup=single/multi,
                time_reduction_percent=100*(1-multi/single),
                note='Same learning workload; excludes initialization, replay warmup, evaluation, checkpoint I/O. GPU rendering and CPU physics overlap; environment_wait is not pure render time.')
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)

if __name__=='__main__':main()
