"""Fixed-base three-ball training; analytic sphere dynamics, physical preparation."""
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
    def __init__(self,basket_mixed_ball='mixed',basket_mixed_options=None,**kwargs):
        if basket_mixed_ball not in ('mixed',)+NAMES:raise ValueError('Unknown Basket mixed ball mode')
        if not (kwargs.get('object_motion') in ('basket','throw_basket') and kwargs.get('task')=='Catching'
                and kwargs.get('basket_fixed_base_training') and kwargs.get('basket_fixed_physical')
                and kwargs.get('basket_carry_reward') and kwargs.get('basket_finger_reward')
                and kwargs.get('basket_reward_version')=='basket_aim_v1'
                and kwargs.get('basket_arm_control')=='joint_velocity'):
            raise ValueError('Three-ball training requires fixed physical Basket finger/carry joint-velocity task')
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
        self.basket_grip_cfg=SimpleNamespace(mobile_hold_hand_flex=self.mixed_options[hand_key],
            mobile_hold_thumb_flex=[.4,.4,.4])
        # Recompute collision bounds and mass constants before resetting/settling.
        mujoco.mj_setConst(model,self.Dcmm.data)

    def _get_obs(self):
        obs=super()._get_obs()
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
        obs,info=super().reset(**kwargs);info.update(self._ball_info());return obs,info

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        info.update(self._ball_info());return obs,reward,done,truncated,info
