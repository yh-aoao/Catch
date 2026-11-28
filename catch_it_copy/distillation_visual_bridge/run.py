"""Visual-only autonomous Basket + multi-target Catch distillation."""
import argparse
import copy
import json
import math
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from distillation.run import atomic_save,save_json,seed_all,wilson
from distillation.teachers import ROOT,sha256
from distillation.core import ActionAdapter
from future_prediction_experiments.training import learning_rate
from .core import SCHEMA,TASKS,Student,Replay,losses,action_mask,ABLATIONS
from .runtime import validate_render_config,render_device,latch_episode_choices,validate_torch_device
from .teachers import FrozenTeacher
from .environments import Pool,terminal_success


def provenance():
    result={}
    for folder in ('distillation_visual_bridge','basket_mobile_bridge','basket_mobile_curriculum','basket_autonomous','distillation_four_task_future','future_prediction_experiments','distillation','distillation_visual_aux',
                   'model_experiments','gym_dcmm','configs','urdf','assets','ycb_objects'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.xml','.urdf','.stl','.obj','.json','.yaml') and not any(x in p.parts for x in ('runs','__pycache__','tests')):
                result[str(p.relative_to(ROOT))]=sha256(p)
    return result


def rng_state(device):
    n=np.random.get_state();device=torch.device(device)
    cuda=None
    if device.type=='cuda':
        cuda=dict(state=torch.cuda.get_rng_state(device),device=str(device))
    return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=(n[0],n[1].tolist(),*n[2:]),cuda=cuda)


def restore_rng(s,saved_device=None,device=None):
    torch.set_rng_state(s['torch']);random.setstate(s['python'])
    n=s['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
    target=torch.device(device or saved_device or 'cpu')
    cuda=s.get('cuda')
    if cuda is None or target.type!='cuda':return
    if isinstance(cuda,dict):state=cuda['state']
    else:
        # Legacy checkpoints stored all visible devices; restore only the active stream.
        source=torch.device(saved_device or target)
        index=source.index if source.index is not None else 0
        if index>=len(cuda):raise ValueError('Checkpoint lacks active CUDA RNG state')
        state=cuda[index]
    torch.cuda.set_rng_state(state.cpu(),target)


def resource_metrics(student,collector):
    result=dict(environment_processes=sum(p.is_alive() for pool in collector.pools.values() for p in pool.processes))
    device=student.mean.device
    if device.type=='cuda':
        result.update(cuda_allocated_mb=torch.cuda.memory_allocated(device)/2**20,
                      cuda_reserved_mb=torch.cuda.memory_reserved(device)/2**20,
                      cuda_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20,
                      cuda_peak_reserved_mb=torch.cuda.max_memory_reserved(device)/2**20)
    return result


def env_config(config,task):
    c=copy.deepcopy(config['tasks'][task]);c.update(stage='catch',vision=config['vision'],extra_labels=False)
    c['env_kwargs']=dict(config['env_kwargs'],**c['env_kwargs'])
    c['render_devices']=[render_device(config,task,i,config['num_envs_per_task']) for i in range(config['num_envs_per_task'])]
    return c


def validate(config):
    validate_render_config(config)
    if config.get('basket_rollin','step') not in ('step','episode'):raise ValueError('Invalid basket_rollin')
    weight=config.get('basket_carry_precision_weight',0.)
    if not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight<0:raise ValueError('Invalid basket_carry_precision_weight')
    if config.get('ablation','full') not in ABLATIONS:raise ValueError('Unknown ablation')
    mode=config.get('sampling_mode','per_task')
    if mode not in ('per_task','per_object'):raise ValueError('Invalid sampling_mode')
    if mode=='per_object' and (type(config.get('batch_episodes_per_object')) is not int or config['batch_episodes_per_object']<1):raise ValueError('Invalid batch_episodes_per_object')
    if tuple(config['tasks'])!=TASKS:raise ValueError('Expected four tasks in canonical order')
    for k in ('iterations','num_envs_per_task','rollout_steps','updates_per_iteration','calibration_steps','eval_every','eval_start',
              'eval_episodes','save_every','batch_episodes_per_object','capacity_frames_per_task','max_episode_steps','bptt_steps'):
        if type(config[k]) is not int or config[k]<1:raise ValueError('Invalid '+k)
    if type(config['evaluation_seed']) is not int or config['evaluation_seed']<0:raise ValueError('Invalid evaluation seed')
    if config['auxiliary']['velocity_weight']!=0:raise ValueError('Position/Future supervision only')
    if not config['skip_teacher_baseline']:raise ValueError('Teacher baseline disabled')
    for task,c in config['tasks'].items():
        if config['num_envs_per_task']<len(c['objects']):raise ValueError('Need one worker per target at minimum')
        if config['capacity_frames_per_task']//len(c['objects'])<config['max_episode_steps']:raise ValueError('Replay too small')
        if not isinstance(c.get('sha256'),str) or len(c['sha256'])!=64 or any(ch not in '0123456789abcdef' for ch in c['sha256']):raise ValueError('Teacher hash not pinned: '+task+'; run distillation_visual_bridge.prepare_config first')
        ActionAdapter(c)
    s=config['lr_schedule']
    if s['kind']!='cosine' or not 0<=s['start']<s['end'] or not 0<s['minimum']<=config['learning_rate']:raise ValueError('Invalid Stable LR schedule')


def student_step(student,packet,previous,hidden):
    device=student.mean.device
    with torch.no_grad():
        values=[torch.as_tensor(packet[k],device=device)[:,None] for k in ('depth','proprio')]
        action,hidden,_=student(*values,torch.as_tensor(previous,device=device)[:,None],
                               torch.as_tensor(packet['context'],device=device)[:,None],hidden)
    return action[:,0].cpu().numpy(),hidden


class Collector:
    def __init__(self,config,student,teachers,adapters,replay,start=0):
        self.config=config;self.student=student;self.teachers=teachers;self.adapters=adapters;self.replay=replay
        self.pools={};self.pending={};self.previous={};self.hidden={};self.states=[]
        self.calibrating=not student.normalizer_fitted.item()
        self.teacher_fraction=1. if self.calibrating else .8
        self.basket_episode_teacher=None
        try:
            for ti,task in enumerate(TASKS):
                n=config['num_envs_per_task'];self.pools[task]=Pool(task,env_config(config,task),n,config['seed']+ti*10000+start*1000000)
                self.pending[task]=[[] for _ in range(n)];self.previous[task]=np.zeros((n,20),np.float32)
                self.hidden[task]=student.initial_state(n)
        except BaseException:self.close();raise

    def collect(self,steps):
        self.student.eval();before=time.monotonic()
        timings=dict(student_inference_seconds=0.,teacher_inference_seconds=0.,environment_wait_seconds=0.)
        for step in range(steps):
            dispatched={}
            for task,pool in self.pools.items():
                packet=pool.obs;stamp=time.monotonic()
                action,hidden=student_step(self.student,packet,self.previous[task],self.hidden[task])
                timings['student_inference_seconds']+=time.monotonic()-stamp
                adapter=self.adapters[task]
                stamp=time.monotonic()
                target=adapter.canonical(self.teachers[task].predict(packet['teacher_obs']))
                timings['teacher_inference_seconds']+=time.monotonic()-stamp
                choose_teacher=np.random.random(len(action))<self.teacher_fraction
                if task=='basket' and self.config.get('basket_rollin','step')=='episode':
                    self.basket_episode_teacher,choose_teacher=latch_episode_choices(choose_teacher,self.basket_episode_teacher)
                action=np.where(choose_teacher[:,None],target,action)
                command=adapter.executable(action)*action_mask(packet['context'])
                for i,pending in enumerate(self.pending[task]):
                    row={k:packet[k][i].copy() for k in ('depth','proprio','context','object_target','object_valid','world_position','basket_target','basket_valid','basket_phase')}
                    row.update(previous=self.previous[task][i].copy(),target=target[i].copy(),contact_after=np.float32(0))
                    pending.append(row)
                if self.calibrating:self.states.append(packet['proprio'].copy())
                objects=packet['object_name'].copy()
                pool.step_async(adapter.action_dict(command))
                dispatched[task]=(objects,command,hidden)
            for task,pool in self.pools.items():
                objects,command,hidden=dispatched[task]
                stamp=time.monotonic()
                _,_,done,infos=pool.step_wait()
                timings['environment_wait_seconds']+=time.monotonic()-stamp
                for i,pending in enumerate(self.pending[task]):
                    pending[-1]['contact_after']=np.float32(infos[i]['module_hand_contact'])
                    if len(pending)>self.config['max_episode_steps']:raise RuntimeError('Episode exceeded configured step limit')
                    if done[i]:
                        self.replay.add(task,str(objects[i]),pending);self.pending[task][i]=[]
                        if task=='basket' and self.basket_episode_teacher is not None:self.basket_episode_teacher[i]=-1
                self.previous[task]=command.copy();self.previous[task][done]=0
                hidden[:,torch.as_tensor(done,device=hidden.device)]=0;self.hidden[task]=hidden
            if steps>=16 and (step+1)%16==0:print('[collect] {}/{} steps/env {:.1f}s'.format(step+1,steps,time.monotonic()-before),flush=True)

        timings['collect_seconds']=time.monotonic()-before
        timings['collect_other_seconds']=max(0.,timings['collect_seconds']-sum(timings[k] for k in ('student_inference_seconds','teacher_inference_seconds','environment_wait_seconds')))
        self.timings=timings
        return timings

    @torch.no_grad()
    def refresh_hidden(self):
        self.student.eval();device=self.student.mean.device
        for task,episodes in self.pending.items():
            for i,rows in enumerate(episodes):
                hidden=self.student.initial_state(1)
                for begin in range(0,len(rows),self.config['bptt_steps']):
                    chunk=rows[begin:begin+self.config['bptt_steps']]
                    values=[torch.as_tensor(np.stack([r[k] for r in chunk]),device=device)[None] for k in ('depth','proprio','previous','context')]
                    _,hidden,_=self.student(*values,hidden)
                self.hidden[task][:,i:i+1]=hidden

    def close(self):
        for pool in self.pools.values():pool.close()


def optimize(student,optimizer,batch,config):
    student.train();optimizer.zero_grad(set_to_none=True);hidden=None;values={}
    device=student.mean.device
    # Depth dominates replay memory. Keep full episodes in host RAM and transfer
    # only the current BPTT chunk; loss metadata is small and stays on the device.
    batch={key:(value if key=='depth' else value.to(device)) for key,value in batch.items()}
    lengths=batch['valid'].sum(1)
    for begin in range(0,batch['depth'].shape[1],config['bptt_steps']):
        end=begin+config['bptt_steps']
        x=[batch[k][:,begin:end].to(device) for k in ('depth','proprio','previous','context')]
        _,hidden,logits,prediction=student(*x,hidden,auxiliary=True)
        terms=losses(student,batch,begin,end,logits,prediction,config,lengths)
        if not torch.isfinite(terms['loss']):raise FloatingPointError('Non-finite loss')
        terms['loss'].backward();hidden=hidden.detach()
        for key,value in terms.items():values[key]=values.get(key,0.)+value.item()
    norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradients')
    optimizer.step();student.extra={}
    return values


@torch.no_grad()
def evaluate(config,adapters,student,output,tasks=None,viewer=False):
    student.eval();results={}
    for task in tasks or TASKS:
        ti=TASKS.index(task);cfg=config['tasks'][task];objects={}
        for oi,obj in enumerate(cfg['objects']):
            count=config['eval_episodes']//len(cfg['objects'])+int(oi<config['eval_episodes']%len(cfg['objects']))
            if count<1:raise ValueError('Evaluation episodes must cover every target')
            n=1 if viewer else min(config.get('eval_num_envs',4),count)
            pool=None;rows=[]
            try:
                pool=Pool(task,env_config(config,task),n,config['evaluation_seed']+1000000+ti*10000+oi*1009,viewer=viewer,objects=[obj])
                previous=np.zeros((n,20),np.float32);hidden=None
                reward_sum=np.zeros(n);length=np.zeros(n,dtype=int)
                for _ in range(count*config['max_episode_steps']+1):
                    command,hidden=student_step(student,pool.obs,previous,hidden)
                    command=adapters[task].executable(command)
                    _,reward,done,infos=pool.step(adapters[task].action_dict(command))
                    reward_sum+=reward;length+=1;previous=command.copy()
                    if (length>config['max_episode_steps']).any():raise RuntimeError('Evaluation step limit')
                    for i in np.flatnonzero(done):
                        if len(rows)<count:
                            info=infos[i];rows.append(dict(episode=len(rows),worker=int(i),success=terminal_success(info,cfg['success_metric']),
                                reward=float(reward_sum[i]),length=int(length[i]),reason=str(info.get('terminated_reason','unspecified')),
                                **({k:bool(info.get(k,False)) for k in ('reached_throw_phase','released','release_stopped','basket_region_hit','basket_stopped_region_hit','safe_basket_success')} if task=='basket' else {})))
                        previous[i]=0;hidden[:,i]=0;reward_sum[i]=0;length[i]=0
                    if len(rows)==count:break
                if len(rows)!=count:raise RuntimeError('Incomplete evaluation')
                successes=sum(r['success'] for r in rows)
                objects[obj]=dict(successes=successes,episodes=count,success_rate=successes/count,wilson_95=wilson(successes,count),records=rows)
            finally:
                if pool is not None:pool.close()
            if task=='basket':
                objects[obj]['diagnostics']={k:sum(r[k] for r in rows) for k in ('reached_throw_phase','released','release_stopped','basket_region_hit','basket_stopped_region_hit','safe_basket_success')}
            print('[eval] {} / {} {}/{}'.format(task,obj,successes,count),flush=True)
        successes=sum(v['successes'] for v in objects.values());episodes=sum(v['episodes'] for v in objects.values())
        results[task]=dict(successes=successes,episodes=episodes,success_rate=successes/episodes,
            object_macro_success_rate=sum(v['success_rate'] for v in objects.values())/len(objects),
            success_metric=cfg['success_metric'],objects=objects,seed=config['evaluation_seed'])
        save_json(output,results)
    return results


def package(student,config,metadata,iteration):
    return dict(schema=SCHEMA,student=student.state_dict(),config=config,metadata=metadata,iteration=iteration)


def should_evaluate(i,c):return (i>=c['eval_start'] and (i-c['eval_start'])%c['eval_every']==0) or i==c['iterations']


def train(config,teachers,adapters,output,resume=None,allow_source_changes=False,init_checkpoint=None):
    student=Student(config).to(config['device']);optimizer=torch.optim.Adam(student.parameters(),lr=config['learning_rate'])
    replay=Replay(config);start=0;best=-float('inf')
    metadata=dict(teacher_hashes={t:v.hash for t,v in teachers.items()},source_asset_sha256=provenance())
    if init_checkpoint is not None:
        initial=torch.load(init_checkpoint,map_location='cpu',weights_only=True)
        from .runtime import validate_initial_student
        validate_initial_student(initial,config,SCHEMA,metadata['teacher_hashes'])
        student.load_state_dict(initial['student'],strict=True)
        save_json(output/'initialization_audit.json',dict(checkpoint=str(init_checkpoint.resolve()),
            checkpoint_sha256=sha256(init_checkpoint),source_iteration=initial['iteration'],
            source_config=initial['config'],new_config=config,
            reset=['optimizer','replay','iteration','teacher_mix_schedule'],
            note='Model/normalization only. New experiment; not an exact resume.'))
        del initial
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        from .runtime import validate_resume
        changes,reset=validate_resume(saved,config,metadata,SCHEMA,allow_source_changes)
        save_json(output/'resume_audit.json',dict(changed_sources=changes,old_config=saved['config'],new_config=config))
        student.load_state_dict(saved['student'],strict=True);optimizer.load_state_dict(saved['optimizer']);replay.load_state_dict(saved['replay'])
        start=saved['iteration'];best=-float('inf') if reset else saved['best_score']
        restore_rng(saved['rng'],saved['config']['device'],config['device'])
    if start>=config['iterations']:raise ValueError('Resume already reached requested iteration')
    collector=Collector(config,student,teachers,adapters,replay,start)
    try:
        if collector.calibrating:
            collector.collect(config['calibration_steps']);student.fit_normalizer(np.concatenate(collector.states))
            collector.states.clear();collector.calibrating=False;collector.refresh_hidden()
        for _ in range(config['max_episode_steps']+1):
            if replay.ready():break
            collector.collect(1)
        if not replay.ready():raise RuntimeError('Warmup did not produce an episode for every condition')
        for iteration in range(start+1,config['iterations']+1):
            before=time.monotonic();lr=learning_rate(config,iteration)
            for g in optimizer.param_groups:g['lr']=lr
            collector.teacher_fraction=max(config['teacher_mix_min'],config['teacher_mix_start']*(1-iteration/config['teacher_mix_decay']))
            timings=collector.collect(config['rollout_steps'])
            stamp=time.monotonic()
            updates=[optimize(student,optimizer,replay.sample(config['batch_episodes_per_object'],'cpu'),config) for _ in range(config['updates_per_iteration'])]
            if student.mean.device.type=='cuda':torch.cuda.synchronize(student.mean.device)
            timings['update_seconds']=time.monotonic()-stamp
            stamp=time.monotonic();collector.refresh_hidden()
            if student.mean.device.type=='cuda':torch.cuda.synchronize(student.mean.device)
            timings['refresh_hidden_seconds']=time.monotonic()-stamp
            row=dict(iteration=iteration,learning_rate=lr,seconds=time.monotonic()-before,teacher_fraction=collector.teacher_fraction,collected_policy_steps=4*config['num_envs_per_task']*config['rollout_steps'],replay_frames=dict(replay.sizes),
                     **timings,**{k:float(np.mean([v[k] for v in updates])) for k in updates[0]})
            stamp=time.monotonic()
            saved=package(student,config,metadata,iteration);atomic_save(output/'student_last.pth',saved)
            save_seconds=time.monotonic()-stamp;row['evaluation_seconds']=0.
            if should_evaluate(iteration,config):
                stamp=time.monotonic()
                atomic_save(output/('student_%06d.pth'%iteration),saved);rng=rng_state(config['device'])
                save_seconds+=time.monotonic()-stamp;stamp=time.monotonic()
                result=evaluate(config,adapters,student,output/('eval_%06d.json'%iteration));restore_rng(rng,config['device'],config['device'])
                row['evaluation_seconds']=time.monotonic()-stamp
                score=min(v['object_macro_success_rate'] for v in result.values());row['min_task_success']=score
                row['macro_success']=sum(v['object_macro_success_rate'] for v in result.values())/4
                if score>best:
                    stamp=time.monotonic();best=score;atomic_save(output/'student_best.pth',saved)
                    save_seconds+=time.monotonic()-stamp
            if iteration%config['save_every']==0 or iteration==config['iterations']:
                stamp=time.monotonic()
                atomic_save(output/'resume_last.pth',dict(saved,optimizer=optimizer.state_dict(),replay=replay.state_dict(),best_score=best,rng=rng_state(config['device'])))
                save_seconds+=time.monotonic()-stamp
            row.update(resource_metrics(student,collector))
            row['save_seconds']=save_seconds;row['seconds']=time.monotonic()-before
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    finally:collector.close()


def env_check(config,teachers,adapters,output,steps):
    """One physical environment per task/object, real rendering; no optimizer updates."""
    student=Student(config).to(config['device']).eval();report={}
    for ti,task in enumerate(TASKS):
        for oi,obj in enumerate(config['tasks'][task]['objects']):
            pool=None
            try:
                pool=Pool(task,env_config(config,task),1,config['seed']+ti*10000+oi*1009,objects=[obj])
                previous=np.zeros((1,20),np.float32);hidden=None
                phases=set();terminals=0
                for _ in range(steps):
                    phases.add(int(pool.obs['context'][0,6]>.5))
                    command,hidden=student_step(student,pool.obs,previous,hidden)
                    target=teachers[task].predict(pool.obs['teacher_obs'])
                    if target.shape!=(1,20) or not np.isfinite(target).all():raise ValueError('Teacher environment inference failed')
                    # Step the teacher for interface smoke only; this is not a success-rate test.
                    command=adapters[task].canonical(target)*action_mask(pool.obs['context'])
                    _,_,done,infos=pool.step(adapters[task].action_dict(command));previous=command
                    if done[0]:
                        terminal_success(infos[0],config['tasks'][task]['success_metric'])
                        terminals+=1;previous.fill(0);hidden=None
                report[task+'/'+obj]=dict(basket_center_in_camera_frustum=pool.obs.get('basket_center_in_camera_frustum',[]).tolist() if task=='basket' else [],depth_shape=list(pool.obs['depth'].shape),proprio_shape=list(pool.obs['proprio'].shape),
                                         context=pool.obs['context'][0].tolist(),teacher_inference_finite=True,
                                         steps=steps,observed_phases=sorted(phases),terminal_metric_checks=terminals)
                print('[env-check] '+task+'/'+obj,flush=True)
            finally:
                if pool is not None:pool.close()
    save_json(output/'environment_check.json',report)


def main():
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','env-check','train','eval'])
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('configs')/'catch.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--init-checkpoint',type=Path,help='Initialize student weights/normalizer in a new experiment; reset optimizer/replay/iteration')
    p.add_argument('--teacher-root',type=Path,help='Directory containing the four unchanged teacher files')
    p.add_argument('--device');p.add_argument('--gl-backend',choices=['egl','glfw','osmesa','auto']);p.add_argument('--egl-device-id',type=int)
    for flag in ('iterations','num-envs','episodes','eval-start','eval-every','seed','eval-seed','rollout-steps'):p.add_argument('--'+flag,type=int)
    p.add_argument('--skip-teacher-baseline',action='store_true');p.add_argument('--allow-source-changes',action='store_true')
    p.add_argument('--render-devices',nargs='+',type=int,help='Round-robin EGL device IDs for environment workers')
    p.add_argument('--render-task-devices',nargs='+',help='Explicit EGL mapping: throw=1 roll=1 bounce=2 basket=3')
    p.add_argument('--ablation',choices=ABLATIONS,help='full (default), no-position (all position heads), no-future, or none')
    p.add_argument('--basket-rollin',choices=['step','episode'],help='Fresh training: Basket action executor selection frequency')
    p.add_argument('--basket-carry-precision-weight',type=float,help='Extra arm/hand action MSE during carry; default 0')
    p.add_argument('--basket-success-metric',choices=['strict','hit'],help='Basket evaluation metric for a fresh run or standalone eval; hit removes stopped-release requirement')
    p.add_argument('--eval-num-envs',type=int,help='Parallel environments per object during evaluation')
    p.add_argument('--check-steps',type=int,default=2)
    p.add_argument('--viewer',action='store_true',help='Open a passive MuJoCo viewer during student evaluation')
    p.add_argument('--tasks',nargs='+',choices=TASKS,help='Evaluation only: select tasks to view/test')
    args=p.parse_args()
    args.output=args.output.expanduser().resolve()
    if args.basket_success_metric and args.command not in ('train','eval'):p.error('--basket-success-metric is train/eval only')
    if args.eval_num_envs is not None and args.eval_num_envs<1:p.error('--eval-num-envs must be positive')
    if (args.viewer or args.tasks) and args.command!='eval':p.error('--viewer and --tasks are evaluation only')
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
    if sum(x is not None for x in (args.egl_device_id,args.render_devices,args.render_task_devices))>1:p.error('Choose one of --egl-device-id, --render-devices, --render-task-devices')
    if args.egl_device_id is not None:
        if args.egl_device_id<0:p.error('EGL device must be nonnegative')
        os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.egl_device_id)
    saved=None
    if args.resume or args.command=='eval':
        source=args.resume if args.resume else args.checkpoint
        if not source:p.error('eval requires --checkpoint')
        saved=torch.load(source,map_location='cpu',weights_only=True)
        if saved.get('schema')!=SCHEMA:raise ValueError('Expected autonomous visual Basket student')
        config=copy.deepcopy(saved['config'])
    else:config=json.loads(args.config.read_text())
    for flag,key,default in [('basket_rollin','basket_rollin','step'),('basket_carry_precision_weight','basket_carry_precision_weight',0.)]:
        value=getattr(args,flag)
        if value is not None:
            if saved is not None and value!=config.get(key,default):p.error('Repair settings cannot change on resume/eval; start a separate run')
            if saved is None:config[key]=value
    if args.ablation is not None:
        if saved is not None and args.ablation!=config.get('ablation','full'):
            p.error('Cannot change ablation when resuming/evaluating a checkpoint; start a new run')
        if saved is None:config['ablation']=args.ablation
    if args.teacher_root:
        for task,cfg in config['tasks'].items():cfg['checkpoint']=str((args.teacher_root/Path(cfg['checkpoint']).name).resolve())
    if args.init_checkpoint and (args.command!='train' or args.resume):p.error('--init-checkpoint is fresh train only, mutually exclusive with --resume')
    if args.resume and args.command!='train':p.error('resume is train only')
    if args.check_steps<1:p.error('check-steps must be positive')
    for flag,key in [('iterations','iterations'),('num_envs','num_envs_per_task'),('episodes','eval_episodes'),('eval_start','eval_start'),('eval_every','eval_every'),('seed','seed'),('eval_seed','evaluation_seed'),('rollout_steps','rollout_steps')]:
        if getattr(args,flag) is not None:config[key]=getattr(args,flag)
    if args.egl_device_id is not None or args.render_devices is not None:
        config['render_devices']=args.render_devices if args.render_devices is not None else [args.egl_device_id]
        config.pop('render_task_devices',None)
    if args.render_task_devices is not None:
        try:
            pairs=[value.split('=') for value in args.render_task_devices]
            mapping={task:int(device) for task,device in pairs}
            if len(mapping)!=len(pairs):raise ValueError('Duplicate task')
        except ValueError:p.error('Expected task=device entries, e.g. throw=1 roll=1 bounce=2 basket=3')
        config['render_task_devices']=mapping;config.pop('render_devices',None)
    if args.gl_backend:config['vision']['gl_backend']=args.gl_backend
    elif args.viewer:config['vision']['gl_backend']='glfw'
    if args.viewer and args.render_devices is None and args.render_task_devices is None and args.egl_device_id is None:
        config.pop('render_devices',None);config.pop('render_task_devices',None)
    config['device']=args.device or ('cpu' if args.command=='check' else config['device'])
    if args.eval_num_envs is not None:config['eval_num_envs']=args.eval_num_envs
    if args.basket_success_metric:
        metric='basket_region_hit' if args.basket_success_metric=='hit' else 'safe_basket_success'
        if args.resume and config['tasks']['basket']['success_metric']!=metric:p.error('Changing the training evaluation metric requires a new run')
        config['tasks']['basket']['success_metric']=metric
    if args.skip_teacher_baseline:config['skip_teacher_baseline']=True
    validate(config)
    config['device']=str(validate_torch_device(config['device']))
    print('[network-device] {} CUDA_VISIBLE_DEVICES={!r}'.format(config['device'],os.environ.get('CUDA_VISIBLE_DEVICES')),flush=True)
    torch.set_num_threads(1);seed_all(config['seed'])
    if args.output.exists() and not args.resume:raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True,exist_ok=bool(args.resume))
    counts=[len(c['objects'])*config['batch_episodes_per_object'] for c in config['tasks'].values()] if config.get('sampling_mode','per_task')=='per_object' else [config.get('batch_episodes_per_task',2)]*4
    print('[sampling] {}: {} = {} sequences/update; task weights each 25%'.format(config.get('sampling_mode','per_task'),'+'.join(map(str,counts)),sum(counts)),flush=True)
    print('[ablation] '+config.get('ablation','full'),flush=True)
    print('[render-devices] '+json.dumps({t:env_config(config,t)['render_devices'] for t in TASKS}),flush=True)
    adapters={t:ActionAdapter(c) for t,c in config['tasks'].items()}
    if args.command=='eval':
        current=provenance();old=saved['metadata']['source_asset_sha256']
        changed={k:dict(training=old.get(k),current=current.get(k)) for k in set(old)|set(current) if old.get(k)!=current.get(k)}
        if changed and not args.allow_source_changes:raise ValueError('Source changed; review before --allow-source-changes')
        save_json(args.output/'evaluation_audit.json',dict(checkpoint_sha256=sha256(args.checkpoint),iteration=saved['iteration'],source_changes=changed,config=config))
        student=Student(config).to(config['device']);student.load_state_dict(saved['student'],strict=True)
        evaluate(config,adapters,student,args.output/'student_evaluation.json',tasks=args.tasks,viewer=args.viewer);return
    teachers={t:FrozenTeacher(t,c,config['device']) for t,c in config['tasks'].items()}
    checks={t:teacher.parity() for t,teacher in teachers.items()};save_json(args.output/'teacher_check.json',checks);print(json.dumps(checks,indent=2),flush=True)
    save_json(args.output/'resolved_config.json',config)
    if args.command=='train':train(config,teachers,adapters,args.output,args.resume,args.allow_source_changes,args.init_checkpoint)
    elif args.command=='env-check':env_check(config,teachers,adapters,args.output,args.check_steps)
    else:
        student=Student(config)
        print('Student: depth + robot36 + previous20 + task1 -> GRU; Basket goal supervision only; no phase action mask',flush=True)


if __name__=='__main__':main()
