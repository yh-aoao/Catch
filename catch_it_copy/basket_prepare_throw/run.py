"""Two-stage PPO: prepare/transport, then throw from actual arrival states."""
import argparse,copy,hashlib,json,os,random,time
from collections import deque,Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def gate_streak_after(level,streak,success_rate):
    return streak+1 if success_rate>=(.8 if level<2 else .7) else 0


def atomic_save(path,value):
    import torch
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);os.replace(temp,path)


def package(model,config,steps,optimizer=None,iteration=0,best=-1.):
    from .policy import SCHEMA
    result=dict(schema=SCHEMA,model=model.state_dict(),config=copy.deepcopy(config),steps=steps,iteration=iteration,
        observation_version='robot36_measured_palmvel_ball6_goal3_previous20_task1_v1',actions='full20_joint_velocity_hand_increment_v1',
        reset_protocol='random_physical_hold_or_ready_replay_v1',reward_version=config['options'].get('reward_version','autonomous_v1'),best=best)
    if optimizer is not None:
        import torch
        result.update(optimizer=optimizer.state_dict(),rng_python=random.getstate(),rng_torch=torch.get_rng_state(),
            rng_numpy=np.random.get_state(),rng_cuda=torch.cuda.get_rng_state(next(model.parameters()).device) if next(model.parameters()).is_cuda else None)
    return result


def evaluate(model,config,count,seed,viewer=False):
    import torch
    device=next(model.parameters()).device;model.eval();records=[]
    if viewer:
        import mujoco.viewer
        from .environment import AutonomousTeacherEnv
        env=AutonomousTeacherEnv(evaluation_options(config));window=None
        try:
            for episode in range(count):
                np.random.seed(seed+episode);obs,_=env.reset(seed=seed+episode)
                if window is None:window=mujoco.viewer.launch_passive(env.Dcmm.model,env.Dcmm.data)
                total=0.
                for step in range(512):
                    started=time.monotonic()
                    action=model.act(torch.as_tensor(obs,device=device)[None])[0].cpu().numpy()
                    obs,reward,terminated,truncated,info=env.step(action);total+=reward
                    window.sync()
                    if not window.is_running():raise KeyboardInterrupt
                    time.sleep(max(0.,.04-(time.monotonic()-started)))
                    if terminated or truncated:break
                records.append(dict(episode=episode,seed=seed+episode,reward=total,steps=step+1,**info))
        finally:
            if window is not None:window.close()
            env.close()
    else:
        from .pool import Pool
        pool=Pool(min(8,count),evaluation_options(config),seed)
        totals=np.zeros(pool.count);lengths=np.zeros(pool.count,dtype=int)
        try:
            for _ in range(count*512):
                action=model.act(torch.as_tensor(pool.obs,device=device)).cpu().numpy()
                _,reward,terminated,truncated,infos,_=pool.step(action);totals+=reward;lengths+=1
                for i in np.flatnonzero(terminated|truncated):
                    if len(records)<count:records.append(dict(episode=len(records),worker=int(i),reward=float(totals[i]),steps=int(lengths[i]),**infos[i]))
                    totals[i]=0;lengths[i]=0
                if len(records)==count:break
        finally:pool.close()
    if len(records)!=count:raise RuntimeError('Incomplete evaluation')
    successes=sum(r['task_success'] for r in records)
    return dict(episodes=count,successes=successes,success_rate=successes/count,reasons=dict(Counter(r['terminated_reason'] for r in records)),records=records)


def train(config,output,resume=None,initialize_from=None):
    import torch
    from .policy import Policy
    from .pool import Pool
    device=torch.device(config['device']);model=Policy().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-5)
    steps=0;iteration=0;best=-1.
    if initialize_from:
        saved=torch.load(initialize_from,map_location='cpu',weights_only=False)
        validate_initialization(saved,config)
        model.load_state_dict(saved['model'],strict=True)
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        steps=saved['steps'];iteration=saved['iteration'];best=saved['best']
        torch.set_rng_state(saved['rng_torch']);random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy'])
        if saved.get('rng_cuda') is not None and device.type=='cuda':torch.cuda.set_rng_state(saved['rng_cuda'],device)
    n=config['num_envs'];pool=Pool(n,config['options'],config['seed']+steps)
    rewards=deque(maxlen=100);success=deque(maxlen=100);totals=np.zeros(n)
    grouped={key:deque(maxlen=200) for key in ('prepare','short_throw','full')}
    gate_streak=int(config.get('gate_streak',0))
    reached=deque(maxlen=200)
    carry_metrics={key:deque(maxlen=100) for key in ('drop_failure','carry_slip_mean','carry_slip_max')}
    next_eval=(steps//config['eval_every']+1)*config['eval_every'];next_save=(steps//config['save_every']+1)*config['save_every']
    try:
        while steps<config['total_steps']:
            started=time.monotonic();iteration+=1;horizon=min(config['rollout_steps'],(config['total_steps']-steps)//n)
            if horizon<1:raise ValueError('total_steps must divide by num_envs')
            model.train();model.update_normalizer(torch.as_tensor(pool.obs,device=device))
            states=[];raw_actions=[];old_logs=[];values=[];deltas=[];dones=[];active_masks=[]
            collection_start=time.monotonic()
            for _ in range(horizon):
                with torch.no_grad():
                    state=model.normalize(torch.as_tensor(pool.obs,device=device))
                    active=torch.ones(n,20,device=device)
                    active[torch.as_tensor(pool.fixed,device=device),:2]=0.
                    action,raw,logp,value=model.sample(state,active)
                obs,reward,terminated,truncated,infos,final=pool.step(action.cpu().numpy())
                with torch.no_grad():next_value=model.value(model.normalize(torch.as_tensor(final,device=device)))
                scaled=torch.as_tensor(reward*config['reward_scale'],dtype=torch.float32,device=device)
                term=torch.as_tensor(terminated,device=device);done=torch.as_tensor(terminated|truncated,device=device)
                active_masks.append(active);states.append(state);raw_actions.append(raw);old_logs.append(logp);values.append(value)
                deltas.append(scaled+config['gamma']*next_value*(~term)-value);dones.append(done)
                totals+=reward
                for i in np.flatnonzero(terminated|truncated):
                    rewards.append(float(totals[i]));success.append(float(infos[i]['task_success']));totals[i]=0.
                    grouped[infos[i]['episode_kind']].append(float(infos[i]['task_success']))
                    if infos[i]['episode_kind']=='full':reached.append(float(infos[i].get('reached_throw_phase',False)))
                    for key,buffer in carry_metrics.items():buffer.append(float(infos[i].get(key,0.)))
                steps+=n
            collection_seconds=time.monotonic()-collection_start
            advantages=torch.zeros(horizon,n,device=device);last=torch.zeros(n,device=device)
            for t in reversed(range(horizon)):
                last=deltas[t]+config['gamma']*config['gae_lambda']*(~dones[t])*last;advantages[t]=last
            returns=(advantages+torch.stack(values)).flatten().detach()
            advantage=advantages.flatten();advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            active=torch.cat(active_masks);state=torch.cat(states);raw=torch.cat(raw_actions);old_log=torch.cat(old_logs);size=len(state)
            metrics=[];early_stop=False
            for epoch in range(config['epochs']):
                order=torch.randperm(size,device=device)
                for start in range(0,size,config['minibatch_size']):
                    idx=order[start:start+config['minibatch_size']]
                    distribution=model.distribution(state[idx]);logp=model.log_probability(distribution,raw[idx],active[idx])
                    ratio=(logp-old_log[idx]).exp()
                    actor=-torch.minimum(ratio*advantage[idx],ratio.clamp(1-config['clip'],1+config['clip'])*advantage[idx]).mean()
                    critic=.5*(model.value(state[idx])-returns[idx]).square().mean()
                    entropy=(distribution.entropy()*active[idx]).sum(-1).mean()
                    loss=actor+config['value_weight']*critic-config['entropy_weight']*entropy
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite PPO loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    with torch.no_grad():kl=((ratio-1)-(logp-old_log[idx])).mean()
                    metrics.append((float(actor),float(critic),float(kl)))
                    if float(kl)>1.5*config['target_kl']:early_stop=True;break
                if early_stop:break
            row=dict(iteration=iteration,steps=steps,collect_seconds=collection_seconds,seconds=time.monotonic()-started,
                physics_fps=horizon*n/max(collection_seconds,1e-6),reward_100=float(np.mean(rewards)) if rewards else None,
                success_100=float(np.mean(success)) if success else None,
                short_throw_sr=float(np.mean(grouped['short_throw'])) if grouped['short_throw'] else None,
                full_sr=float(np.mean(grouped['full'])) if grouped['full'] else None,
                actor_loss=float(np.mean([v[0] for v in metrics])),value_loss=float(np.mean([v[1] for v in metrics])),kl=float(np.mean([v[2] for v in metrics])))
            row['bridge_reached_sr']=float(np.mean(reached)) if reached else None
            row['prepare_sr']=float(np.mean(grouped['prepare'])) if grouped['prepare'] else None
            row['preparation_level']=config['options'].get('preparation_level',0)
            row.update({key+'_100':float(np.mean(buffer)) if buffer else None for key,buffer in carry_metrics.items()})
            if steps>=next_eval or steps==config['total_steps']:
                before_python=random.getstate();before_numpy=np.random.get_state();before_torch=torch.get_rng_state()
                result=evaluate(model,config,config['eval_episodes'],config['seed']+500000)
                random.setstate(before_python);np.random.set_state(before_numpy);torch.set_rng_state(before_torch)
                (output/('eval_%09d.json'%steps)).write_text(json.dumps(result,indent=2))
                row['eval_success_rate']=result['success_rate']
                rank=result['success_rate']+(config['options'].get('preparation_level',0) if config['options']['two_stage']=='prepare' else 0.)
                if rank>best:best=rank;atomic_save(output/'teacher_best.pth',package(model,config,steps,best=best,iteration=iteration))
                atomic_save(output/('teacher_%09d.pth'%steps),package(model,config,steps,best=best,iteration=iteration))
                level=config['options'].get('preparation_level',0)
                if config['options']['two_stage']=='prepare' and level<2:
                    gate_streak=gate_streak_after(0,gate_streak,result['success_rate'])
                    config['gate_streak']=gate_streak
                    if gate_streak>=2:
                        pool.close();config['options']['preparation_level']=level+1
                        config['gate_streak']=0;gate_streak=0
                        pool=Pool(n,config['options'],config['seed']+steps)
                        totals=np.zeros(n);rewards.clear();success.clear()
                        for buffer in grouped.values():buffer.clear()
                        reached.clear()
                        atomic_save(output/('promotion_level%d.pth'%(level+1)),package(model,config,steps,best=best,iteration=iteration))
                next_eval=(steps//config['eval_every']+1)*config['eval_every']
            if steps>=next_save or steps==config['total_steps']:
                atomic_save(output/'teacher_last.pth',package(model,config,steps,best=best,iteration=iteration))
                atomic_save(output/'resume_last.pth',package(model,config,steps,optimizer=optimizer,best=best,iteration=iteration))
                next_save=(steps//config['save_every']+1)*config['save_every']
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    finally:pool.close()


def evaluation_options(config):
    options=copy.deepcopy(config['options'])
    options['evaluation_task']='prepare' if options['two_stage']=='prepare' else 'full'
    return options


def validate_initialization(saved,config):
    if saved.get('schema')!='basket_prepare_throw_v1':raise ValueError('Expected new prepare/throw stage-one checkpoint')
    source=saved['config']['options'];target=config['options']
    if source.get('two_stage')!='prepare' or target.get('two_stage')!='throw':
        raise ValueError('Only prepare -> throw stage initialization is supported')
    if source.get('preparation_level',0)!=2:raise ValueError('Finish stage-one distance curriculum before stage two')
    for key in ('pose_bank_entries','arm_jitter_rad','hand_jitter_rad','arm_velocity_protocol','release_protocol'):
        if source.get(key)!=target.get(key):raise ValueError('Stage initialization protocol differs: '+key)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','eval'])
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/stage1_prepare.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--initialize-from',type=Path,help='Initialize stage two from stage-one prepare weights; reset optimizer/steps')
    p.add_argument('--ready-bank',type=Path);p.add_argument('--device');p.add_argument('--pose-bank',type=Path);p.add_argument('--viewer',action='store_true')
    for name in ('num-envs','total-steps','rollout-steps','eval-every','eval-episodes','save-every','seed'):p.add_argument('--'+name,type=int)
    args=p.parse_args()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['MUJOCO_GL']='glfw' if args.viewer else 'egl'
    source=args.resume or args.checkpoint
    if args.command=='eval' and not args.checkpoint:p.error('eval requires --checkpoint')
    if args.command=='train' and (args.checkpoint or args.viewer):p.error('train uses --resume for continuation, not --checkpoint/viewer')
    if args.command=='eval' and args.resume:p.error('--resume is train only')
    if args.initialize_from and (args.command!='train' or source):p.error('--initialize-from is train only and cannot combine with resume/checkpoint')
    if args.initialize_from:args.initialize_from=args.initialize_from.resolve()
    if source:source=source.resolve()
    args.output=args.output.resolve()
    import torch
    from .policy import Policy,SCHEMA
    if source:
        saved=torch.load(source,map_location='cpu',weights_only=False)
        if saved.get('schema')!=SCHEMA:raise ValueError('Expected new autonomous teacher checkpoint; old phase-based teachers are incompatible')
        config=copy.deepcopy(saved['config'])
    else:config=json.loads(args.config.resolve().read_text())
    os.chdir(ROOT)
    if args.pose_bank:
        if source:p.error('Cannot change pose bank when resuming/evaluating')
        config['options']['pose_bank']=str(args.pose_bank.resolve())
    bank=config['options'].pop('pose_bank',None)
    if bank:
        path=Path(bank);config['options']['pose_bank_entries']=json.loads(path.read_text())
        config['pose_bank_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('num_envs','total_steps','rollout_steps','eval_every','eval_episodes','save_every','seed'):
        value=getattr(args,name)
        if value is not None:
            if args.resume and name in ('seed','num_envs'):p.error('Resume preserves seed and num_envs')
            config[name]=value
    config['device']=args.device or config['device']
    for key in ('num_envs','total_steps','rollout_steps','eval_every','eval_episodes','save_every','epochs','minibatch_size'):
        if type(config[key]) is not int or config[key]<1:raise ValueError('Invalid '+key)
    if config['total_steps']%config['num_envs']:raise ValueError('total_steps must be divisible by num_envs')
    if config['device'].startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if args.ready_bank:
        config['options']['ready_bank']=str(args.ready_bank.resolve())
    if args.command=='train' and config['options']['two_stage']=='throw':
        bank_path=Path(config['options'].get('ready_bank',''))
        if not bank_path.is_file():raise ValueError('Stage two requires --ready-bank')
        digest=hashlib.sha256(bank_path.read_bytes()).hexdigest()
        if args.resume and config.get('ready_bank_sha256')!=digest:raise ValueError('Resume ready bank differs')
        config['ready_bank_sha256']=digest
    if args.initialize_from:
        initial=torch.load(args.initialize_from,map_location='cpu',weights_only=False)
        validate_initialization(initial,config)
        import pickle
        bank=pickle.loads(Path(config['options']['ready_bank']).read_bytes())
        if bank['source_checkpoint_sha256']!=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest():
            raise ValueError('Ready bank must be collected from the initialization checkpoint')
        config['initialization']=dict(path=str(args.initialize_from),sha256=hashlib.sha256(args.initialize_from.read_bytes()).hexdigest())
    elif args.command=='train' and not args.resume and config['options']['two_stage']=='throw':
        p.error('Stage two requires --initialize-from new preparation checkpoint (or --resume)')
    if args.output.exists() and not args.resume:raise FileExistsError('Choose a new output directory')
    args.output.mkdir(parents=True,exist_ok=bool(args.resume))
    torch.set_num_threads(1);torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    (args.output/'resolved_config.json').write_text(json.dumps(config,indent=2))
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')}
    for rel in ('basket_autonomous/environment.py','basket_autonomous/carry_reward.py','gym_dcmm/envs/DcmmVecEnv.py'):
        hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    (args.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    if args.command=='train':train(config,args.output,args.resume,args.initialize_from)
    else:
        model=Policy().to(config['device']);model.load_state_dict(saved['model'],strict=True)
        result=evaluate(model,config,config['eval_episodes'],config['seed']+500000,args.viewer)
        (args.output/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
