"""Conditional fixed-base throws and single-ball mobile training with shared sphere physics."""
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium.spaces as spaces
import mujoco
import numpy as np

from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv

NAMES=('original','tennis_ball','baseball')
PRIORS=Path(__file__).resolve().parents[2]/'ycb_objects/calibration_data/balls.roll_table_priors_v1.json'
DEFAULTS=dict(weights=[1.,1.,1.],original_hand=[.8,.7,.7],
              tennis_hand=[1.,.9,.9],baseball_hand=[.8,.7,.7])


def make_options(options=None):
    options=dict(options or {})
    if set(options)-set(DEFAULTS):raise ValueError('Unknown Basket mixed options')
    result=dict(DEFAULTS,**options)
    for key,value in result.items():
        array=np.asarray(value,dtype=float)
        if array.shape!=(3,) or not np.isfinite(array).all():raise ValueError(key+' must have three finite values')
        if key=='weights' and (np.any(array<0) or array.sum()<=0):raise ValueError('Ball weights must be nonnegative with positive sum')
        result[key]=array.tolist()
    return result


class BasketMixedEnv(DcmmVecEnv):
    def __init__(self,basket_mixed_ball='mixed',basket_mixed_options=None,
                 basket_delivery_version='legacy',basket_delivery_options=None,**kwargs):
        if basket_mixed_ball not in ('mixed',)+NAMES:raise ValueError('Unknown Basket mixed ball mode')
        mobile=kwargs.get('basket_reward_version')=='basket_mobile_v1'
        self.single_mobile=mobile
        if mobile and not (kwargs.get('object_motion') in ('basket','throw_basket')
                and kwargs.get('task')=='Catching' and not kwargs.get('basket_fixed_base_training')
                and kwargs.get('basket_fixed_physical') and kwargs.get('basket_carry_reward')
                and kwargs.get('basket_finger_reward') and kwargs.get('basket_arm_control')=='joint_velocity'):
            raise ValueError('Single mobile requires physical finger/carry mobile Basket')
        if mobile and basket_mixed_ball not in ('tennis_ball','baseball'):
            raise ValueError('Mobile single-ball requires tennis_ball or baseball')
        if not mobile and not (kwargs.get('object_motion') in ('basket','throw_basket') and kwargs.get('task')=='Catching'
                and kwargs.get('basket_fixed_base_training') and kwargs.get('basket_fixed_physical')
                and kwargs.get('basket_carry_reward') and kwargs.get('basket_finger_reward')
                and kwargs.get('basket_reward_version')=='basket_aim_v1'
                and kwargs.get('basket_arm_control')=='joint_velocity'):
            raise ValueError('Three-ball training requires fixed physical Basket finger/carry joint-velocity task')
        from gym_dcmm.utils.basket_delivery import make_options as delivery_options,new_state
        if basket_delivery_version not in ('legacy','ballistic_v2'):
            raise ValueError('Invalid Basket delivery reward version')
        self.delivery_version=basket_delivery_version
        self.delivery_options=delivery_options(basket_delivery_options)
        self.delivery_state=new_state()
        self.ball_mode=basket_mixed_ball
        self.mixed_options=make_options(basket_mixed_options)
        self.profiles=json.loads(PRIORS.read_text())['profiles']
        self.ball_name='original';self.ball_index=0
        super().__init__(**kwargs)
        model=self.Dcmm.model
        self._ball_body=int(model.geom_bodyid[self.object_id])
        joint=int(model.body_jntadr[self._ball_body]);dof=int(model.jnt_dofadr[joint])
        self._ball_dofs=slice(dof,dof+6)
        # Original is the exact trained protocol, not the different table prior.
        self._original=dict(size=model.geom_size[self.object_id].copy(),
            friction=model.geom_friction[self.object_id].copy(),condim=int(model.geom_condim[self.object_id]),
            rgba=model.geom_rgba[self.object_id].copy(),mass=float(model.body_mass[self._ball_body]),
            inertia=model.body_inertia[self._ball_body].copy(),ipos=model.body_ipos[self._ball_body].copy(),
            iquat=model.body_iquat[self._ball_body].copy(),damping=model.dof_damping[self._ball_dofs].copy(),
            armature=model.dof_armature[self._ball_dofs].copy())
        self.mixed_protocol=dict(version='mixed37_known_radius_mass_v1',
            original={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self._original.items()},
            tennis_ball=self.profiles['tennis_ball'],baseball=self.profiles['baseball'])
        if mobile:
            return # Ball is constant; policy uses the existing mobile51 interface.
        self.observation_space.spaces['basket_ball_properties']=spaces.Box(0.,np.inf,shape=(2,),dtype=np.float32)
        self.observation_space.spaces['basket_ball_id']=spaces.Box(0.,2.,shape=(1,),dtype=np.float32)
        # Physical properties enter policy; categorical identity is logging metadata only.
        self.obs_dim+=2;self.obs_c_dim+=2;self.obs_t_dim+=2
        print('[basket-three-ball] obs=37 actor=arm18; known radius/mass; mode='+self.ball_mode,flush=True)

    def _reset_simulation(self):
        if self.ball_mode=='mixed':
            weights=np.asarray(self.mixed_options['weights'],dtype=float)
            self.ball_index=int(self.np_random.choice(3,p=weights/weights.sum()))
        else:self.ball_index=NAMES.index(self.ball_mode)
        self.ball_name=NAMES[self.ball_index]
        self._apply_ball()
        super()._reset_simulation()

    def _apply_ball(self):
        model=self.Dcmm.model;geom=self.object_id;body=self._ball_body;original=self._original
        if self.ball_name=='original':
            radius=original['size'][0];mass=original['mass']
            model.geom_size[geom]=original['size'];model.geom_friction[geom]=original['friction']
            model.geom_condim[geom]=original['condim'];model.geom_rgba[geom]=original['rgba']
            model.body_inertia[body]=original['inertia'];model.body_ipos[body]=original['ipos']
            model.body_iquat[body]=original['iquat'];model.dof_damping[self._ball_dofs]=original['damping']
            model.dof_armature[self._ball_dofs]=original['armature']
        else:
            profile=self.profiles[self.ball_name];radius=profile['radius_m'];mass=profile['mass_kg']
            model.geom_size[geom]=[radius,0.,0.]
            model.geom_friction[geom]=[profile['sliding_friction'],profile['torsional_length_m'],profile['rolling_length_m']]
            model.geom_condim[geom]=6
            model.geom_rgba[geom]=[.7,.9,.1,1.] if self.ball_name=='tennis_ball' else [.95,.95,.95,1.]
            model.body_inertia[body]=profile['inertia_factor']*mass*radius**2
            model.body_ipos[body]=0.;model.body_iquat[body]=[1.,0.,0.,0.]
            model.dof_damping[self._ball_dofs]=0.;model.dof_armature[self._ball_dofs]=0.
        model.body_mass[body]=mass
        self.random_mass=float(mass)
        self.basket_cfg.basket_ball_radius=float(radius);self.basket_cfg.basket_ball_mass=float(mass)
        hand_key=('original_hand','tennis_hand','baseball_hand')[self.ball_index]
        if self.single_mobile:
            self.basket_cfg.mobile_hold_hand_flex=self.mixed_options[hand_key]
        self.basket_grip_cfg=SimpleNamespace(mobile_hold_hand_flex=self.mixed_options[hand_key],
            mobile_hold_thumb_flex=[.4,.4,.4])
        # Recompute collision bounds and mass constants before resetting/settling.
        mujoco.mj_setConst(model,self.Dcmm.data)

    def _get_obs(self):
        obs=super()._get_obs()
        if self.single_mobile:
            return obs
        obs['basket_ball_properties']=np.array([self.basket_cfg.basket_ball_radius,
            self.basket_cfg.basket_ball_mass],dtype=np.float32)
        obs['basket_ball_id']=np.array([self.ball_index],dtype=np.float32)
        return obs

    def _ball_info(self):
        return dict(basket_ball=self.ball_name,basket_ball_index=self.ball_index,
            basket_ball_radius=float(self.basket_cfg.basket_ball_radius),
            basket_ball_mass=float(self.basket_cfg.basket_ball_mass),
            basket_ball_protocol='mixed37_known_radius_mass_v1')

    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs)
        from gym_dcmm.utils.basket_delivery import new_state,quality
        q,_=quality(self.Dcmm.data.qpos[37:40],self.Dcmm.data.qvel[36:39],
            self.basket_center,self.Dcmm.model.opt.gravity,self.basket_cfg.basket_tilt_deg,self.delivery_options)
        self.delivery_state=new_state(q)
        info.update(self._ball_info());return obs,info

    def _basket_check_flight(self,previous):
        super()._basket_check_flight(previous)
        if self.delivery_version!='ballistic_v2':return
        from gym_dcmm.utils.basket_delivery import quality,observe
        from gym_dcmm.utils.roll_rewards import hand_collision_ids
        hands=hand_collision_ids(self.Dcmm.model,self.hand_start_id)
        contacts=self.contacts['object_contacts']
        touching=bool(np.any(np.isin(contacts,hands)))
        other=bool(np.any(~np.isin(contacts,hands)))
        unsafe=bool(self.basket_arm_self_collision or self.basket_robot_collision or other)
        launch=self.basket_simple_state
        active=not self.single_mobile or self.basket_mobile_state.get('unlocked',False)
        if touching and active and not launch.get('confirmed') and not unsafe:
            q,error=quality(self.Dcmm.data.qpos[37:40],self.Dcmm.data.qvel[36:39],
                self.basket_center,self.Dcmm.model.opt.gravity,self.basket_cfg.basket_tilt_deg,self.delivery_options)
            observe(self.delivery_state,q,error,self.delivery_options)
        if launch.get('confirmed') and 'release_quality' not in self.delivery_state:
            candidate=launch['candidate']
            q,error=quality(candidate['position'],candidate['velocity'],self.basket_center,
                self.Dcmm.model.opt.gravity,self.basket_cfg.basket_tilt_deg,self.delivery_options)
            self.delivery_state.update(release_quality=q,release_error=-1. if error is None else error)

    def compute_reward(self,obs,info,ctrl):
        value=super().compute_reward(obs,info,ctrl)
        if self.delivery_version!='ballistic_v2':return value
        from gym_dcmm.utils.basket_delivery import adjustments
        terms=info['basket_reward_terms'];old=terms.copy()
        collision=bool(info.get('robot_dangerous_collision',False))
        deadline=self.basket_cfg.basket_max_time
        if self.single_mobile:
            from gym_dcmm.utils.basket_mobile import episode_deadline
            deadline=episode_deadline(self.basket_mobile_state,self.basket_cfg)
        done=bool(self.terminated or info['env_time']>deadline)
        # ballistic_v2 already charges a terminal miss; avoid a second mobile miss cost.
        if self.single_mobile:
            terms['task_failure']=0.
        terms.update(adjustments(self.delivery_state,old,self.delivery_options,
            done=done,success=bool(info.get('safe_basket_success',False)),collision=collision))
        for key,amount in terms.items():
            delta=amount-old.get(key,0.)
            value+=delta
            self.basket_reward_totals[key]=self.basket_reward_totals.get(key,0.)+delta
        info['basket_reward_totals']=self.basket_reward_totals.copy()
        state=self.delivery_state
        info['basket_control'].update(reward_revision='basket_delivery_v2',
            ballistic_quality=state.get('quality',0.),ballistic_best=state.get('best',0.),
            release_ballistic_quality=state.get('release_quality',0.),
            release_ballistic_error=state.get('release_error',-1.))
        return value

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        info.update(self._ball_info());return obs,reward,done,truncated,info
