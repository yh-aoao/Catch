"""Four-task Future-Stable distillation: check / env-check / train / eval."""
import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from distillation.run import atomic_save,save_json,seed_all,wilson
from distillation.teachers import ROOT,sha256
from distillation.core import ActionAdapter
from future_prediction_experiments.training import learning_rate
from .core import SCHEMA,TASKS,Student,Replay,losses,action_mask
from .teachers import FrozenTeacher
from .environments import Pool,terminal_success


def provenance():
    result={}
    for folder in ('distillation_four_task_future','future_prediction_experiments','distillation','distillation_visual_aux',
                   'model_experiments','gym_dcmm','configs','urdf','assets','ycb_objects'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.xml','.urdf','.stl','.obj','.json','.yaml') and not any(x in p.parts for x in ('runs','__pycache__','tests')):
                result[str(p.relative_to(ROOT))]=sha256(p)
    return result


def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=(n[0],n[1].tolist(),*n[2:]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s,saved_device=None,device=None):
    torch.set_rng_state(s['torch']);random.setstate(s['python'])
    n=s['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
    from model_experiments.visual_resume import restore_cuda_rng
    restore_cuda_rng(s['cuda'],saved_device,device)


def env_config(config,task):
    c=copy.deepcopy(config['tasks'][task]);c.update(stage='catch',vision=config['vision'],extra_labels=False)
    c['env_kwargs']=dict(config['env_kwargs'],**c['env_kwargs'])
    return c


def validate(config):
    if tuple(config['tasks'])!=TASKS:raise ValueError('Expected four tasks in canonical order')
    for k in ('iterations','num_envs_per_task','rollout_steps','updates_per_iteration','calibration_steps','eval_every','eval_start',
              'eval_episodes','save_every','batch_episodes_per_object','capacity_frames_per_task','max_episode_steps','bptt_steps'):
        if type(config[k]) is not int or config[k]<1:raise ValueError('Invalid '+k)
    if config['eval_episodes']%3:raise ValueError('Evaluation total per task must divide equally over 3 balls')
    if config['auxiliary']['velocity_weight']!=0:raise ValueError('This version uses position/Future supervision only')
    if not config['skip_teacher_baseline']:raise ValueError('Teacher closed-loop baseline is disabled in this experiment')
    if config['vision']['gl_backend'] not in ('egl','auto','glfw','osmesa'):raise ValueError('Invalid renderer')
    if config['num_envs_per_task']<3:raise ValueError('Need >=3 envs/task to cover all balls')
    for task,c in config['tasks'].items():
        expected=['original'] if task in ('throw','basket') else ['original','tennis_ball','baseball']
        if c['objects']!=expected:raise ValueError('Unexpected object protocol')
        if config['capacity_frames_per_task']//len(expected)<config['max_episode_steps']:raise ValueError('Replay must fit full episodes per object')
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
        try:
            for ti,task in enumerate(TASKS):
                n=config['num_envs_per_task'];self.pools[task]=Pool(task,env_config(config,task),n,config['seed']+ti*10000+start*1000000)
                self.pending[task]=[[] for _ in range(n)];self.previous[task]=np.zeros((n,20),np.float32)
                self.hidden[task]=student.initial_state(n)
        except BaseException:self.close();raise

    def collect(self,steps):
        self.student.eval();before=time.monotonic()
        for step in range(steps):
            for task,pool in self.pools.items():
                packet=pool.obs;action,hidden=student_step(self.student,packet,self.previous[task],self.hidden[task])
                adapter=self.adapters[task]
                target=adapter.canonical(self.teachers[task].predict(packet['teacher_obs']))
                command=adapter.executable(action)*action_mask(packet['context'])
                for i,pending in enumerate(self.pending[task]):
                    row={k:packet[k][i].copy() for k in ('depth','proprio','context','object_target','object_valid','world_position')}
                    row.update(previous=self.previous[task][i].copy(),target=target[i].copy(),contact_after=np.float32(0))
                    pending.append(row)
                if self.calibrating:self.states.append(packet['proprio'].copy())
                objects=packet['object_name'].copy()
                _,_,done,infos=pool.step(adapter.action_dict(command))
                for i,pending in enumerate(self.pending[task]):
                    pending[-1]['contact_after']=np.float32(infos[i]['module_hand_contact'])
                    if len(pending)>self.config['max_episode_steps']:raise RuntimeError('Episode exceeded configured step limit')
                    if done[i]:
                        self.replay.add(task,str(objects[i]),pending);self.pending[task][i]=[]
                self.previous[task]=command.copy();self.previous[task][done]=0
                hidden[:,torch.as_tensor(done,device=hidden.device)]=0;self.hidden[task]=hidden
            if steps>=16 and (step+1)%16==0:print('[collect] {}/{} steps/env {:.1f}s'.format(step+1,steps,time.monotonic()-before),flush=True)

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
    lengths=batch['valid'].sum(1)
    for begin in range(0,batch['depth'].shape[1],config['bptt_steps']):
        end=begin+config['bptt_steps']
        x=[batch[k][:,begin:end] for k in ('depth','proprio','previous','context')]
        _,hidden,logits,prediction=student(*x,hidden,auxiliary=True)
        terms=losses(student,batch,begin,end,logits,prediction,config,lengths)
        if not torch.isfinite(terms['loss']):raise FloatingPointError('Non-finite loss')
        terms['loss'].backward();hidden=hidden.detach()
        for key,value in terms.items():values[key]=values.get(key,0.)+value.item()
    norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradients')
    optimizer.step();return values


@torch.no_grad()
def evaluate(config,adapters,student,output):
    student.eval();results={}
    for ti,task in enumerate(TASKS):
        objects={};cfg=config['tasks'][task];count=config['eval_episodes']//len(cfg['objects'])
        for oi,obj in enumerate(cfg['objects']):
            pool=None;rows=[];reward_sum=0.;length=0
            try:
                pool=Pool(task,env_config(config,task),1,config['evaluation_seed']+1000000+ti*10000+oi*1009,objects=[obj])
                previous=np.zeros((1,20),np.float32);hidden=None
                for _ in range(count*config['max_episode_steps']+1):
                    command,hidden=student_step(student,pool.obs,previous,hidden)
                    command=adapters[task].executable(command)*action_mask(pool.obs['context'])
                    _,reward,done,infos=pool.step(adapters[task].action_dict(command))
                    reward_sum+=float(reward[0]);length+=1;previous=command.copy()
                    if length>config['max_episode_steps']:raise RuntimeError('Evaluation episode exceeded step limit')
                    if done[0]:
                        info=infos[0];rows.append(dict(episode=len(rows),success=terminal_success(info,cfg['success_metric']),reward=reward_sum,
                                                     length=length,reason=str(info.get('terminated_reason','unspecified')),
                                                     stable_hold=bool(info.get('roll_stable_hold',False))))
                        previous.fill(0);hidden=None;reward_sum=0.;length=0
                        if len(rows)%25==0:print('[eval] {} / {} {}/{}'.format(task,obj,len(rows),count),flush=True)
                        if len(rows)==count:break
                if len(rows)!=count:raise RuntimeError('Incomplete evaluation')
                successes=sum(r['success'] for r in rows)
                objects[obj]=dict(successes=successes,episodes=count,success_rate=successes/count,wilson_95=wilson(successes,count),records=rows)
            finally:
                if pool is not None:pool.close()
        successes=sum(v['successes'] for v in objects.values());episodes=sum(v['episodes'] for v in objects.values())
        results[task]=dict(successes=successes,episodes=episodes,success_rate=successes/episodes,
                           object_macro_success_rate=sum(v['success_rate'] for v in objects.values())/len(objects),
                           success_metric=cfg['success_metric'],objects=objects,seed=config['evaluation_seed'])
        save_json(output,results)
        print('[eval] {} {:.2%}'.format(task,results[task]['success_rate']),flush=True)
    return results


def package(student,config,metadata,iteration):
    return dict(schema=SCHEMA,student=student.state_dict(),config=config,metadata=metadata,iteration=iteration)


def should_evaluate(i,c):return (i>=c['eval_start'] and (i-c['eval_start'])%c['eval_every']==0) or i==c['iterations']


def train(config,teachers,adapters,output,resume=None,allow_source_changes=False):
    student=Student(config).to(config['device']);optimizer=torch.optim.Adam(student.parameters(),lr=config['learning_rate'])
    replay=Replay(config);start=0;best=-float('inf')
    metadata=dict(teacher_hashes={t:v.hash for t,v in teachers.items()},source_asset_sha256=provenance())
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        from model_experiments.visual_resume import validate_resume
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
            collector.collect(config['rollout_steps'])
            updates=[optimize(student,optimizer,replay.sample(config['batch_episodes_per_object'],config['device']),config) for _ in range(config['updates_per_iteration'])]
            collector.refresh_hidden()
            row=dict(iteration=iteration,learning_rate=lr,seconds=time.monotonic()-before,replay_frames=dict(replay.sizes),
                     **{k:float(np.mean([v[k] for v in updates])) for k in updates[0]})
            saved=package(student,config,metadata,iteration);atomic_save(output/'student_last.pth',saved)
            if should_evaluate(iteration,config):
                atomic_save(output/('student_%06d.pth'%iteration),saved);rng=rng_state()
                result=evaluate(config,adapters,student,output/('eval_%06d.json'%iteration));restore_rng(rng)
                score=min(v['object_macro_success_rate'] for v in result.values());row['min_task_success']=score
                row['macro_success']=sum(v['object_macro_success_rate'] for v in result.values())/4
                if score>best:best=score;atomic_save(output/'student_best.pth',saved)
            if iteration%config['save_every']==0 or iteration==config['iterations']:
                atomic_save(output/'resume_last.pth',dict(saved,optimizer=optimizer.state_dict(),replay=replay.state_dict(),best_score=best,rng=rng_state()))
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
                for _ in range(steps):
                    command,hidden=student_step(student,pool.obs,previous,hidden)
                    target=teachers[task].predict(pool.obs['teacher_obs'])
                    if target.shape!=(1,20) or not np.isfinite(target).all():raise ValueError('Teacher environment inference failed')
                    # Step the teacher for interface smoke only; this is not a success-rate test.
                    command=adapters[task].canonical(target)*action_mask(pool.obs['context'])
                    _,_,done,_=pool.step(adapters[task].action_dict(command));previous=command
                    if done[0]:previous.fill(0);hidden=None
                report[task+'/'+obj]=dict(depth_shape=list(pool.obs['depth'].shape),proprio_shape=list(pool.obs['proprio'].shape),
                                         context=pool.obs['context'][0].tolist(),teacher_inference_finite=True)
                print('[env-check] '+task+'/'+obj,flush=True)
            finally:
                if pool is not None:pool.close()
    save_json(output/'environment_check.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','env-check','train','eval'])
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('configs')/'catch.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--device');p.add_argument('--gl-backend',choices=['egl','glfw','osmesa','auto']);p.add_argument('--egl-device-id',type=int)
    for flag in ('iterations','num-envs','episodes','eval-start','eval-every','seed'):p.add_argument('--'+flag,type=int)
    p.add_argument('--skip-teacher-baseline',action='store_true');p.add_argument('--allow-source-changes',action='store_true')
    p.add_argument('--check-steps',type=int,default=2)
    args=p.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
    if args.egl_device_id is not None:
        if args.egl_device_id<0:p.error('EGL device must be nonnegative')
        os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.egl_device_id)
    saved=None
    if args.resume or args.command=='eval':
        source=args.resume if args.resume else args.checkpoint
        if not source:p.error('eval requires --checkpoint')
        saved=torch.load(source,map_location='cpu',weights_only=True)
        if saved.get('schema')!=SCHEMA:raise ValueError('Expected four-task student, not an older Future student')
        config=copy.deepcopy(saved['config'])
    else:config=json.loads(args.config.read_text())
    if args.resume and args.command!='train':p.error('resume is train only')
    if args.check_steps<1:p.error('check-steps must be positive')
    for flag,key in [('iterations','iterations'),('num_envs','num_envs_per_task'),('episodes','eval_episodes'),('eval_start','eval_start'),('eval_every','eval_every'),('seed','seed')]:
        if getattr(args,flag) is not None:config[key]=getattr(args,flag)
    if args.gl_backend:config['vision']['gl_backend']=args.gl_backend
    config['device']=args.device or ('cpu' if args.command=='check' else config['device'])
    if args.skip_teacher_baseline:config['skip_teacher_baseline']=True
    validate(config)
    if config['device'].startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);seed_all(config['seed'])
    if args.output.exists() and not args.resume:raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True,exist_ok=bool(args.resume))
    adapters={t:ActionAdapter(c) for t,c in config['tasks'].items()}
    if args.command=='eval':
        current=provenance();old=saved['metadata']['source_asset_sha256']
        changed={k:dict(training=old.get(k),current=current.get(k)) for k in set(old)|set(current) if old.get(k)!=current.get(k)}
        if changed and not args.allow_source_changes:raise ValueError('Source changed; review before --allow-source-changes')
        save_json(args.output/'evaluation_audit.json',dict(checkpoint_sha256=sha256(args.checkpoint),iteration=saved['iteration'],source_changes=changed,config=config))
        student=Student(config).to(config['device']);student.load_state_dict(saved['student'],strict=True)
        evaluate(config,adapters,student,args.output/'student_evaluation.json');return
    teachers={t:FrozenTeacher(t,c,config['device']) for t,c in config['tasks'].items()}
    checks={t:teacher.parity() for t,teacher in teachers.items()};save_json(args.output/'teacher_check.json',checks);print(json.dumps(checks,indent=2),flush=True)
    save_json(args.output/'resolved_config.json',config)
    if args.command=='train':train(config,teachers,adapters,args.output,args.resume,args.allow_source_changes)
    elif args.command=='env-check':env_check(config,teachers,adapters,args.output,args.check_steps)
    else:
        student=Student(config)
        print('Student: depth + robot36 + previous20 + context10 -> GRU -> catch/Basket20; from scratch',flush=True)


if __name__=='__main__':main()
