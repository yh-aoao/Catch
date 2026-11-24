"""No task-phase actuator gating, scripted hold, braking, or in-episode ball actuation."""
import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
import gymnasium.spaces as spaces
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from gym_dcmm.utils.basket_catching import hoop_crossing
from gym_dcmm.utils.basket_aim import crossing_error
from gym_dcmm.utils.basket_grip import palm_coordinates
from gym_dcmm.utils.basket_simple import arm_self_contacts
from gym_dcmm.utils.roll_rewards import hand_collision_ids
from .carry_reward import VERSION as CARRY_VERSION,carry_terms,region_distance,navigation_progress,purposeful_launch

SCHEMA='basket_autonomous_teacher_v1'
HAND=np.array([0,2,3,4,6,7,8,10,11,13,14,15])


class AutonomousTeacherEnv(DcmmVecEnv):
    basket_autonomous_control=True
    observation_version='robot36_ball6_goal3_previous20_v1'
    def __init__(self,options=None,**kwargs):
        self.options=dict(options or {})
        self.training_stage=self.options.get('training_stage','joint')
        if self.training_stage not in ('joint','fixed','mobile'):
            raise ValueError('Unknown autonomous training stage')
        probability=float(self.options.get('fixed_replay_probability',0.))
        if not 0.<=probability<=1.:raise ValueError('Invalid fixed replay probability')
        self.reward_version=self.options.get('reward_version','autonomous_v1')
        if self.reward_version not in ('autonomous_v1',CARRY_VERSION):raise ValueError('Unknown autonomous reward version')
        self.arm_velocity_protocol=self.options.get('arm_velocity_protocol','measured_v1')
        self.release_protocol=self.options.get('release_protocol','contact_gap_v1')
        if self.arm_velocity_protocol not in ('measured_v1','integrated_target_v2'):raise ValueError('Unknown velocity protocol')
        if self.release_protocol not in ('contact_gap_v1','separated_flight_v2'):raise ValueError('Unknown release protocol')
        self.basket_integrated_arm_velocity=self.arm_velocity_protocol=='integrated_target_v2'
        self.fixed_episode=False
        self.radius=.033275785333333335;self.mass=.058
        self.reset_stats={};self.last_action=np.zeros(20,np.float32)
        defaults=dict(task='Catching',object_motion='throw_basket',viewer=False,render_mode=None,
            render_per_step=False,img_size=(160,160),imshow_cam=False,object_eval=False,
            env_time=self.options.get('episode_seconds',10.),steps_per_policy=20,
            basket_arm_control='joint_velocity',basket_reward_version='basket_mobile_v1',
            basket_fixed_base_training=False,basket_fixed_physical=False,
            basket_carry_reward=False,basket_finger_reward=False,
            basket_mobile_options=dict(target_x_range=[-.2,.2],target_y_range=[1.15,1.55],
                preserve_fixed_throw=False,fixed_throw_probability=0.,post_ready_seconds=0.,
                hold_position=False,approach_braking=False,brake_after_unlock=False,
                carry_settle_seconds=.001,max_speed=.3,max_acceleration=.35,max_time=self.options.get('episode_seconds',10.)))
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.basket_cfg.basket_ball_radius=self.radius
        self.basket_cfg.basket_ball_mass=self.mass
        if self.basket_fixed_base_training or self.basket_control_probe!='off':raise ValueError('Autonomous task cannot use base fixation or controller probes')
        self._base_render=self.render
        self.render=lambda:np.empty(0,np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(65,),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,(20,),dtype=np.float32)
        from configs.env import DcmmCfg
        cup=np.zeros(16)
        for first in (0,4,8):cup[[first,first+2,first+3]]=[1.,.9,.9]
        cup[13:16]=.4
        self.pose_bank=[dict(arm=np.asarray(DcmmCfg.arm_joints).tolist(),hand=cup.tolist(),local=[0.,.03,.07],source='generic_cup_neighborhood')]
        bank=self.options.get('pose_bank')
        entries=self.options.get('pose_bank_entries',[])
        if bank:entries=json.loads(Path(bank).read_text())
        for item in entries:
            if np.shape(item['arm'])!=(6,) or np.shape(item['hand'])!=(16,) or np.shape(item['local'])!=(3,):raise ValueError('Invalid pose-bank entry')
            self.pose_bank.append(item)
        self._initializing=False

    def _reset_object_throw_basket(self):
        root=ET.fromstring(super()._reset_object_throw_basket())
        body=root.find(".//body[@name='object']")
        for old in body.findall('inertial'):body.remove(old)
        inertia=.55*self.mass*self.radius**2
        ET.SubElement(body,'inertial',dict(pos='0 0 0',mass=str(self.mass),diaginertia=' '.join([str(inertia)]*3)))
        geom=body.find("geom[@name='object']")
        geom.set('size',str(self.radius));geom.set('friction','0.5 0.0001 0.0003');geom.set('condim','6')
        geom.set('rgba','0.7 0.9 0.1 1')
        joint=body.find('joint');joint.set('damping','0');joint.set('armature','0')
        # Physical rim; score still requires ball clearance through the opening.
        for rim in root.findall("./worldbody/body[@name='basket_target']/geom"):
            rim.set('contype','1');rim.set('conaffinity','1')
        self.random_mass=self.mass
        return ET.tostring(root,encoding='unicode')

    def touching(self):
        ids=set(hand_collision_ids(self.Dcmm.model,self.hand_start_id).tolist())
        return any(int(g) in ids for g in self._get_contacts()['object_contacts'])

    def _clip_joints(self,start,values):
        values=np.asarray(values,float).copy()
        for i in range(len(values)):
            jid=int(np.flatnonzero(self.Dcmm.model.jnt_qposadr==start+i)[0])
            if self.Dcmm.model.jnt_limited[jid]:values[i]=np.clip(values[i],*self.Dcmm.model.jnt_range[jid])
        return values

    def _reset_targets(self):
        d=self.Dcmm.data
        self.Dcmm.target_base_vel[:]=0.
        self.Dcmm.target_arm_qpos[:]=d.qpos[15:21]
        self.Dcmm.target_hand_qpos[:]=d.qpos[21:37]
        self.Dcmm.data_arm.qpos[:6]=d.qpos[15:21]
        mujoco.mj_fwdPosition(self.Dcmm.model_arm,self.Dcmm.data_arm)
        for buffer in self.action_buffer.values():buffer.clear()
        for name in ('steer_pid','drive_pid','arm_pid','hand_pid'):
            pid=getattr(self.Dcmm,name);pid.integral[:]=0.;pid.e_prev[:]=0.;pid.init=False
        self.update_target_ctrl()

    def reset(self,*,seed=None,options=None):
        self._initializing=True
        super().reset(seed=seed,options=options)
        m,d=self.Dcmm.model,self.Dcmm.data
        base_state=d.qpos.copy();start_time=float(d.time)
        hands=set(hand_collision_ids(m,self.hand_start_id).tolist())
        attempts=int(self.options.get('reset_attempts',80))
        for attempt in range(attempts):
            mujoco.mj_resetData(m,d);d.qpos[:]=base_state;d.time=start_time
            anchor=self.pose_bank[int(self.np_random.integers(len(self.pose_bank)))]
            arm_scale=np.asarray(self.options.get('arm_jitter_rad',[.35,.20,.25,.22,.25,.30]))
            d.qpos[15:21]=self._clip_joints(15,np.asarray(anchor['arm'])+self.np_random.uniform(-arm_scale,arm_scale))
            hand_scale=float(self.options.get('hand_jitter_rad',.20))
            d.qpos[21:37]=self._clip_joints(21,np.asarray(anchor['hand'])+self.np_random.uniform(-hand_scale,hand_scale,16))
            d.qpos[:2]+=self.np_random.uniform(-.08,.08,2)
            d.qvel[:]=0.;mujoco.mj_forward(m,d)
            local=np.asarray(anchor['local'])+self.np_random.uniform([-.008,-.008,-.005],[.008,.008,.005])
            d.qpos[37:40]=d.body('link6').xpos+d.body('link6').xmat.reshape(3,3)@local
            d.qpos[40:44]=[1.,0.,0.,0.];mujoco.mj_forward(m,d)
            # Reset-only collision-free placement. No in-episode lift or correction.
            from gym_dcmm.utils.basket_hold import clear_spawn
            try:placement=clear_spawn(m,d,self.object_id,np.array(sorted(hands)))
            except ValueError:continue
            self._reset_targets()
            stable=0.;valid=True
            # Only reset preparation: random robot targets stay still, ball evolves under contact/gravity.
            for _ in range(int(.45/m.opt.timestep)):
                d.ctrl[:-1]=self._get_ctrl();d.ctrl[-1]=0.
                mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
                contacts=self._get_contacts();touch=any(int(g) in hands for g in contacts['object_contacts'])
                if (any(int(g) not in hands for g in contacts['object_contacts']) or
                    contacts['base_contacts'].size or arm_self_contacts(m,d,1.)):
                    valid=False;break
                local=palm_coordinates(d,d.qpos[37:40])
                held=touch and np.linalg.norm(d.qvel[36:39])<.35 and np.linalg.norm(local)<.18
                stable=stable+m.opt.timestep if held else 0.
                if d.time-start_time>=.12 and stable>=.04:break
            if valid and stable>=.04:break
        else:raise RuntimeError('No physically supported random hold found; widen pose bank or inspect sampling (no fixed fallback)')
        self.reset_stats=dict(attempts=attempt+1,anchor=anchor.get('source','pose_bank'),
            arm=d.qpos[15:21].tolist(),hand=d.qpos[21:37].tolist(),ball_local=palm_coordinates(d,d.qpos[37:40]).tolist(),
            preparation_seconds=float(d.time-start_time),placement=placement)
        self.fixed_episode=(self.training_stage=='fixed' or
            (self.training_stage=='mobile' and self.np_random.random()<float(self.options.get('fixed_replay_probability',0.))))
        if self.fixed_episode:
            # The fixed-base curriculum is explicit; mobile episodes never use this zero command.
            x=self.options.get('fixed_goal_x_range',[-.10,.10])
            y=self.options.get('fixed_goal_y_range',[.85,1.15])
            if len(x)!=2 or len(y)!=2 or x[0]>x[1] or not 0.<y[0]<=y[1]:
                raise ValueError('Invalid fixed goal ranges')
            base=d.body('arm_base')
            local=np.array([self.np_random.uniform(*x),self.np_random.uniform(*y),0.])
            self.basket_center[:2]=(base.xpos+base.xmat.reshape(3,3)@local)[:2]
            m.body_pos[m.body('basket_target').id]=self.basket_center
            mujoco.mj_forward(m,d)
        self.steps=0
        self.start_time=float(d.time);self.terminated=False;self.terminated_reason=None
        self.basket_arm_self_collision=False;self.basket_robot_collision=False
        self._reset_targets();self.last_action.fill(0)
        self._no_hand_seconds=0.;self._release=None;self._last_touch=None;self._low_speed_time=0.
        self._scored=False;self._release_reward=0.;self._quality_best=0.;self._released_once=False
        self._initializing=False
        self.prev_ee_pos3d=self._get_relative_ee_pos3d().copy();self.prev_obj_pos3d=self._get_relative_object_pos3d().copy()
        self._initial_ball=d.qpos[37:40].copy()
        self._ever_intentional_launch=False
        if self.release_protocol=='separated_flight_v2':
            self._last_touch=dict(position=d.qpos[37:40].copy(),velocity=d.qvel[36:39].copy(),
                time=float(d.time),base_speed=float(np.linalg.norm(d.qvel[:2])),stopped_seconds=0.)
        self._carry_slip_integral=0.;self._carry_exposure=0.;self._carry_slip_max=0.
        self._navigation_best=self._navigation_distance()
        self._reward_totals={}
        return self.observation(),dict(reset_sampling=copy.deepcopy(self.reset_stats),fixed_episode=self.fixed_episode)

    def _navigation_distance(self):
        d=self.Dcmm.data;base=d.body('arm_base')
        goal=base.xmat.reshape(3,3).T@(self.basket_center-base.xpos)
        return region_distance(goal,self.options.get('fixed_goal_x_range',[-.1,.1]),
            self.options.get('fixed_goal_y_range',[.85,1.15]))

    def _carry_measurements(self):
        d=self.Dcmm.data;palm=d.body('link6');rotation=palm.xmat.reshape(3,3)
        twist=np.zeros(6)
        mujoco.mj_objectVelocity(self.Dcmm.model,d,mujoco.mjtObj.mjOBJ_BODY,
            self.Dcmm.model.body('link6').id,twist,0)
        ball=d.qpos[37:40];velocity=d.qvel[36:39]
        local=rotation.T@(ball-palm.xpos)
        relative=rotation.T@(velocity-twist[3:]-np.cross(twist[:3],ball-palm.xpos))
        direction=self.basket_center[:2]-ball[:2]
        direction/=max(float(np.linalg.norm(direction)),1e-9)
        base=d.body('arm_base');base_twist=np.zeros(6)
        mujoco.mj_objectVelocity(self.Dcmm.model,d,mujoco.mjtObj.mjOBJ_BODY,
            self.Dcmm.model.body('arm_base').id,base_twist,0)
        palm_carrier=base_twist[3:]+np.cross(base_twist[:3],palm.xpos-base.xpos)
        ball_carrier=base_twist[3:]+np.cross(base_twist[:3],ball-base.xpos)
        # Throw-intent fade excludes carrier motion: driving with a stable hand is not a swing.
        return local,relative,float(np.dot((twist[3:]-palm_carrier)[:2],direction)),float(np.dot((velocity-ball_carrier)[:2],direction))

    def observation(self):
        d=self.Dcmm.data
        ee=self._get_relative_ee_pos3d()
        ee_velocity=(ee-self.prev_ee_pos3d)/(self.steps_per_policy*self.Dcmm.model.opt.timestep)
        self.prev_ee_pos3d=ee.copy()
        base=d.body('arm_base');rotation=base.xmat.reshape(3,3)
        ball=d.qpos[37:40];velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.Dcmm.model,d,mujoco.mjtObj.mjOBJ_BODY,self.Dcmm.model.body('arm_base').id,velocity,0)
        robot=np.r_[self._get_base_vel(),ee,self._get_relative_ee_quat(),ee_velocity,self._get_hand_obs(),d.qpos[15:21],d.qvel[14:20]]
        relative_velocity=rotation.T@(d.qvel[36:39]-velocity[3:]-np.cross(velocity[:3],ball-base.xpos))
        out=np.r_[robot,rotation.T@(ball-base.xpos),relative_velocity,rotation.T@(self.basket_center-base.xpos),self.last_action].astype(np.float32)
        if out.shape!=(65,) or not np.isfinite(out).all():raise FloatingPointError('Invalid autonomous teacher observation')
        return out

    def _hand_clearance(self,hands):
        # Minimum surface distance to ALL hand collision geometries, not palm-center distance.
        return min(float(mujoco.mj_geomDistance(self.Dcmm.model,self.Dcmm.data,
            self.object_id,int(g),1.,None)) for g in hands)

    def _update_confirmed_release(self,touch,p,speed,dt,hands):
        d=self.Dcmm.data
        if touch:
            self._no_hand_seconds=0.
            self._last_touch=dict(position=p.copy(),velocity=d.qvel[36:39].copy(),time=float(d.time),
                base_speed=speed,stopped_seconds=self._low_speed_time)
            # Short contact gaps and recontacts cannot earn release credit or disable carrying.
            self._release=None
            return
        self._no_hand_seconds+=dt
        if self._last_touch is None or self._no_hand_seconds<.04:return
        if self._release is None:
            self._release=copy.deepcopy(self._last_touch)
            r=self._release
            r['predicted_error']=crossing_error(r['position'],r['velocity'],self.basket_center,
                self.Dcmm.model.opt.gravity,self.basket_cfg.basket_tilt_deg)
            r['valid_stop']=r['stopped_seconds']>=.12
            r['distance']=float(np.linalg.norm(r['position'][:2]-self.basket_center[:2]))
            r['confirmed']=False
        r=self._release
        moved=float(np.linalg.norm(p-r['position']))
        if (not r['confirmed'] and self._no_hand_seconds>=.10 and moved>=.20
                and self._hand_clearance(hands)>=.01):
            r['confirmed']=True
            intentional=purposeful_launch(r['position'],r['velocity'],self.basket_center,r['predicted_error'])
            self._ever_intentional_launch |= intentional
            if not self._released_once:
                error=r['predicted_error'];quality=0. if error is None else math.exp(-(error/.5)**2)
                self._release_reward=8.*quality if r['valid_stop'] and intentional else -2.
                self._released_once=True

    def _basket_check_flight(self,previous):
        if self._initializing:return
        d=self.Dcmm.data;m=self.Dcmm.model;dt=float(m.opt.timestep)
        contacts=self._get_contacts();hands=set(hand_collision_ids(m,self.hand_start_id).tolist())
        touch=any(int(g) in hands for g in contacts['object_contacts']);p=d.qpos[37:40].copy()
        speed=float(np.linalg.norm(d.qvel[:2]))
        self._low_speed_time=self._low_speed_time+dt if speed<.05 and np.linalg.norm(d.qvel[3:6])<.15 else 0.
        if self.release_protocol=='separated_flight_v2':
            self._update_confirmed_release(touch,p,speed,dt,hands)
        else:
            if touch:
                self._no_hand_seconds=0.
                self._last_touch=dict(position=p.copy(),velocity=d.qvel[36:39].copy(),time=float(d.time),base_speed=speed,
                    stopped_seconds=self._low_speed_time)
                # Recontact is physical; no regrasp, teleport or target hold is applied.
                self._release=None
            else:
                self._no_hand_seconds+=dt
                if self._last_touch is not None and self._no_hand_seconds>=.04 and self._release is None:
                    self._release=copy.deepcopy(self._last_touch)
                    error=crossing_error(self._release['position'],self._release['velocity'],self.basket_center,m.opt.gravity,self.basket_cfg.basket_tilt_deg)
                    self._release['predicted_error']=error
                    self._ever_intentional_launch |= purposeful_launch(self._release['position'],self._release['velocity'],self.basket_center,error)
                    self._release['valid_stop']=self._release['stopped_seconds']>=.12
                    self._release['distance']=float(np.linalg.norm(self._release['position'][:2]-self.basket_center[:2]))
                    if not self._released_once:
                        quality=0. if error is None else math.exp(-(error/.5)**2)
                        self._release_reward=8.*quality if self._release['valid_stop'] else -2.
                        self._released_once=True
        floor=any(int(g)==self.floor_id for g in contacts['object_contacts'])
        crossed,scored,_=hoop_crossing(previous,p,self.basket_center,self.basket_cfg.basket_tilt_deg,self.basket_cfg.basket_radius,self.radius)
        release=self._release
        airborne=bool(release and d.time-release['time']>=.1 and np.linalg.norm(p-release['position'])>=.2)
        if crossed and scored and airborne and not touch and (self.release_protocol=='contact_gap_v1' or release.get('confirmed',False)) and release['valid_stop'] and release['distance']>=.3 and not floor:
            self._scored=True;self.terminated=True;self.terminated_reason='basket_score'
        elif floor or p[2]<self.radius*.8:
            self.terminated=True;self.terminated_reason='ball_on_floor'
        elif np.linalg.norm(p[:2]-self.basket_center[:2])>4.:
            self.terminated=True;self.terminated_reason='out_of_bounds'

    def step(self,action):
        command=np.asarray(action,np.float32)
        if command.shape!=(20,) or not np.isfinite(command).all():raise ValueError('Expected finite full20 normalized action')
        command=np.clip(command,-1,1)
        self.steps+=1
        previous_action=self.last_action.copy()
        executed=command.copy()
        if self.fixed_episode:executed[:2]=0.
        physical=executed*np.repeat([.3,.025,.15],[2,6,12])
        self._step_mujoco_simulation(dict(base=physical[:2],arm=physical[2:8],hand=physical[8:]))
        d=self.Dcmm.data
        self.last_action=executed.copy();touch=self.touching()
        elapsed=float(d.time-self.start_time)
        truncated=elapsed>=float(self.options.get('episode_seconds',10.)) and not self.terminated
        if truncated:self.terminated_reason='timeout'
        error=crossing_error(d.qpos[37:40],d.qvel[36:39],self.basket_center,self.Dcmm.model.opt.gravity,self.basket_cfg.basket_tilt_deg)
        quality=0. if error is None else math.exp(-(error/.5)**2)
        progress=max(0.,quality-self._quality_best);self._quality_best=max(self._quality_best,quality)
        dt=self.steps_per_policy*self.Dcmm.model.opt.timestep
        terms=dict(time=-.02,contact=.005*float(touch),ballistic_progress=4.*progress,
            release=self._release_reward,action_cost=-.002*float(np.mean(executed**2)),
            action_smoothness=-.01*float(np.mean((executed-previous_action)**2)),score=0.,failure=0.)
        self._release_reward=0.
        drop_failure=bool(self.terminated_reason in ('ball_on_floor','out_of_bounds') and not self._ever_intentional_launch)
        diagnostics={}
        if self.reward_version==CARRY_VERSION:
            # Replace the unconditional contact bonus; stability is weaker than the time cost.
            terms['contact']=0.
            local,relative,palm_forward,ball_forward=self._carry_measurements()
            carry,diagnostics=carry_terms(local,relative,touch,self._released_once,palm_forward,ball_forward,dt)
            terms.update(carry)
            eligible=(not self.fixed_episode and diagnostics['carry_gate']>.5 and
                diagnostics['in_support_envelope'] and diagnostics['slip_speed']<.25)
            self._navigation_best,terms['carry_navigation']=navigation_progress(
                self._navigation_best,self._navigation_distance(),eligible)
            terms['drop_failure']=-5.*float(drop_failure)
            if diagnostics['carry_gate']>0.:
                exposure=dt*diagnostics['carry_gate']
                self._carry_exposure+=exposure;self._carry_slip_integral+=exposure*diagnostics['slip_speed']
                self._carry_slip_max=max(self._carry_slip_max,diagnostics['slip_speed'])
        safe_score=self._scored and not self.basket_robot_collision and not self.basket_arm_self_collision
        if safe_score:terms['score']=100.
        elif self.terminated or truncated:terms['failure']=-15.
        reward=sum(terms.values())
        for key,value in terms.items():self._reward_totals[key]=self._reward_totals.get(key,0.)+value
        info=dict(fixed_episode=self.fixed_episode,success=bool(self._scored),safe_basket_success=bool(self._scored and not self.basket_robot_collision and not self.basket_arm_self_collision),
            terminated_reason=self.terminated_reason,hand_contact=touch,elapsed=elapsed,
            base_speed=float(np.linalg.norm(d.qvel[:2])),low_speed_seconds=self._low_speed_time,
            released=self._released_once,release_stopped=bool(self._release and self._release['valid_stop']),
            predicted_error=error,reset_attempts=self.reset_stats['attempts'],reward_version=self.reward_version,arm_velocity_protocol=self.arm_velocity_protocol,release_protocol=self.release_protocol,
            release_pending=bool(self._release and not self._release.get('confirmed',True)),
            reward_terms=terms,reward_totals=self._reward_totals.copy(),carry_diagnostics=diagnostics,
            carry_seconds=self._carry_exposure,carry_slip_mean=self._carry_slip_integral/max(self._carry_exposure,1e-9),
            carry_slip_max=self._carry_slip_max,drop_failure=drop_failure,purposeful_launch=self._ever_intentional_launch)
        return self.observation(),float(reward),bool(self.terminated),bool(truncated),info
