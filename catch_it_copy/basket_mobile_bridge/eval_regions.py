"""Evaluate enlarged virtual receiving circles on unchanged bridge trajectories."""
import argparse,copy,json,math,os,time
from pathlib import Path
import numpy as np


def circle_hit(radial,diameter,ball_radius):
    return radial is not None and radial<=diameter/2.-ball_radius


def summarize(records,diameters,ball_radius):
    n=len(records)
    regions=[]
    for diameter in sorted(set(diameters)):
        hits=sum(r['safe'] and circle_hit(r['crossing_radial'],diameter,ball_radius) for r in records)
        stopped=sum(r['safe'] and r['release_stopped'] and circle_hit(r['crossing_radial'],diameter,ball_radius) for r in records)
        regions.append(dict(diameter_m=diameter,hits=hits,success_rate=hits/n if n else None,
            stopped_hits=stopped,stopped_success_rate=stopped/n if n else None))
    original=sum(r['original_success'] for r in records)
    return dict(episodes=n,original_successes=original,original_success_rate=original/n if n else None,
        valid_launches=sum(r['valid_launch'] for r in records),regions=regions,
        smallest_tested_diameter_at_least_80pct=next((r['diameter_m'] for r in regions if n and r['success_rate']>=.8),None))


def make_env(options):
    from .environment import AutonomousTeacherEnv
    from gym_dcmm.utils.basket_catching import hoop_crossing
    from basket_autonomous.carry_reward import purposeful_launch
    class ReceivingEnv(AutonomousTeacherEnv):
        def __init__(self,options):
            self.receiving_events=[]
            super().__init__(options)
        def reset(self,**kwargs):
            self.receiving_events=[]
            return super().reset(**kwargs)
        def _basket_check_flight(self,previous):
            super()._basket_check_flight(previous)
            if self._initializing:return
            release=self._release
            if not (self._released_once and release and release.get('confirmed',False) and not self.touching()):return
            intentional=purposeful_launch(release['position'],release['velocity'],self.basket_center,release['predicted_error'])
            if not intentional:return
            crossed,_,radial=hoop_crossing(previous,self.Dcmm.data.qpos[37:40],self.basket_center,
                self.basket_cfg.basket_tilt_deg,0.,0.)
            if crossed:self.receiving_events.append(dict(radial=float(radial),stopped=bool(release['valid_stop']),
                time=float(self.Dcmm.data.time-self.start_time)))
    return ReceivingEnv(options)


def draw_circles(viewer,env,diameters):
    import mujoco
    colors=([1.,.3,.2,.8],[.2,1.,.3,.8],[.3,.6,1.,.8],[1.,.8,.2,.8],[.9,.3,1.,.8])
    tilt=math.radians(env.basket_cfg.basket_tilt_deg)
    u=np.array([1.,0.,0.]);v=np.array([0.,math.cos(tilt),math.sin(tilt)])
    scene=viewer.user_scn;scene.ngeom=0
    for index,diameter in enumerate(sorted(set(diameters))):
        for segment in range(48):
            if scene.ngeom>=scene.maxgeom:return
            a=2*math.pi*segment/48.;b=2*math.pi*(segment+1)/48.
            start=env.basket_center+diameter/2.*(u*math.cos(a)+v*math.sin(a))
            end=env.basket_center+diameter/2.*(u*math.cos(b)+v*math.sin(b))
            geom=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(colors[index%len(colors)]))
            mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,2.,start,end)
            scene.ngeom+=1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=100);p.add_argument('--diameters',type=float,nargs='+',default=[.6,.8,1.,1.2,1.5])
    p.add_argument('--seed',type=int,default=900123);p.add_argument('--device',default='cpu')
    p.add_argument('--viewer',action='store_true');p.add_argument('--speed',type=float,default=1.)
    args=p.parse_args()
    if args.episodes<1 or not np.isfinite(args.diameters).all() or min(args.diameters)<=0 or not math.isfinite(args.speed) or args.speed<=0:
        p.error('episodes, diameters and speed must be positive and finite')
    if args.output.exists():raise FileExistsError('Choose a new output directory: '+str(args.output))
    os.environ['MUJOCO_GL']='glfw' if args.viewer else 'egl'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    import torch,hashlib
    from .policy import Policy,SCHEMA
    from .run import evaluation_options
    torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved.get('schema')!=SCHEMA:raise ValueError('Expected basket_mobile_bridge_v1 checkpoint')
    options=evaluation_options(saved['config']);model=Policy().to(args.device)
    model.load_state_dict(saved['model'],strict=True);model.eval()
    args.output.mkdir(parents=True)
    metadata=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        requested_episodes=args.episodes,seed=args.seed,diameters_m=args.diameters,evaluation_options=options,
        crossing_selection='First eligible descending crossing, not best-of-multiple crossings.',
        geometry='Circular virtual areas in the ORIGINAL tilted hoop plane; descending crossings; full ball clearance required.',
        metric='Confirmed purposeful physical release, no dangerous robot collision. Stopped release also reported separately.',
        unchanged_physical_rim=True,unchanged_episode_termination=True)
    (args.output/'config.json').write_text(json.dumps(metadata,indent=2))
    env=make_env(options);window=None;records=[];interrupted=False
    try:
        if args.viewer:
            import mujoco.viewer
        for episode in range(args.episodes):
            seed=args.seed+episode;np.random.seed(seed);torch.manual_seed(seed)
            obs,_=env.reset(seed=seed)
            if args.viewer and window is None:window=mujoco.viewer.launch_passive(env.Dcmm.model,env.Dcmm.data)
            for step in range(512):
                started=time.monotonic()
                with torch.no_grad():action=model.act(torch.as_tensor(obs,device=args.device)[None])[0].cpu().numpy()
                obs,_,done,truncated,info=env.step(action)
                if window:
                    with window.lock():draw_circles(window,env,args.diameters)
                    window.sync()
                    if not window.is_running():raise KeyboardInterrupt
                    time.sleep(max(0.,.04/args.speed-(time.monotonic()-started)))
                if done or truncated:break
            else:raise RuntimeError('Episode exceeded 512 steps')
            events=env.receiving_events
            best=events[0] if events else None
            record=dict(episode=episode,seed=seed,reason=info['terminated_reason'],original_success=bool(info['safe_basket_success']),
                valid_launch=bool(info.get('purposeful_launch') and info.get('released')),
                safe=not (env.basket_robot_collision or env.basket_arm_self_collision),
                crossing_radial=best['radial'] if best else None,release_stopped=best['stopped'] if best else False,
                base_displacement=info.get('base_displacement'),elapsed=info['elapsed'],events=copy.deepcopy(events))
            records.append(record)
            with (args.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
    except KeyboardInterrupt:interrupted=True
    finally:
        if window is not None:window.close()
        radius=env.radius;env.close()
        summary=summarize(records,args.diameters,radius)
        summary.update(complete=len(records)==args.episodes,interrupted=interrupted)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
