"""Single MuJoCo scene and ball, logical handoff only; never reset at handoff."""
import copy
import hashlib
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
from gym_dcmm.utils.basket_grip import palm_coordinates
from distillation_four_task_future.environments import hand_contact


def physical_snapshot(env):
    model,data=env.Dcmm.model,env.Dcmm.data
    fields={k:np.asarray(getattr(data,k)).copy() for k in ('qpos','qvel','ctrl','act')}
    obj=model.body('object').id
    fields.update(time=np.asarray(data.time),object_mass=np.asarray(model.body_mass[obj]),
                  object_inertia=model.body_inertia[obj].copy(),geom_size=model.geom_size.copy(),
                  geom_friction=model.geom_friction.copy(),geom_solref=model.geom_solref.copy())
    return fields


def snapshot_hash(fields):
    digest=hashlib.sha256()
    for key,array in sorted(fields.items()):digest.update(key.encode());digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


class StableHold:
    def __init__(self,seconds=.3,relative_speed=.3,palm_distance=.18):
        if min(seconds,relative_speed,palm_distance)<=0:raise ValueError('Hold thresholds must be positive')
        self.seconds=seconds;self.relative_speed=relative_speed;self.palm_distance=palm_distance;self.elapsed=0.
    def update(self,dt,touching,speed,distance,safe=True):
        valid=bool(safe and touching and math.isfinite(speed) and math.isfinite(distance)
                   and speed<=self.relative_speed and distance<=self.palm_distance)
        self.elapsed=self.elapsed+dt if valid else 0.
        return self.elapsed+1e-9>=self.seconds


class CatchThenBasketEnv(DcmmVecEnv):
    """Construct a Basket-capable scene, reset as Throw, preserve physics at transition."""
    def __init__(self,basket_kwargs,catch_kwargs,hold_seconds=.3,relative_speed=.3,palm_distance=.18):
        self.hold_detector=StableHold(hold_seconds,relative_speed,palm_distance)
        self.catch_timeout=float(catch_kwargs.get('env_time',2.5))
        self.chain_phase='catch';self.transition=None;self.max_hold_seconds=0.
        self._basket_flags=dict(mobile_basket=True,basket_fixed_physical=True,basket_carry_reward=True,
                                basket_finger_reward=False,basket_arm_control='joint_velocity')
        kwargs=dict(basket_kwargs);kwargs.update(task='Catching',object_motion='throw_basket',viewer=False,
                                               render_mode=None,render_per_step=False,imshow_cam=False)
        super().__init__(**kwargs)
        # Capture the original teacher's parking offset once at reset, not at the moved catch pose.
        self._nominal_front_distance=None
        self._catch_flags=dict(mobile_basket=False,basket_fixed_physical=False,basket_carry_reward=False,
                               basket_finger_reward=False,basket_arm_control='cartesian')
        self.render=lambda:np.empty((0,),np.float32)

    def _reset_object_throw_basket(self):
        # Called by the original constructor. Keep Throw ball dynamics and copy ONLY the hoop.
        mode=self.object_motion
        self.object_motion='throw'
        throw_xml=super()._reset_object_throw();mass=self.random_mass
        self.object_motion=mode
        basket_xml=super()._reset_object_throw_basket();self.random_mass=mass
        root=ET.fromstring(throw_xml);donor=ET.fromstring(basket_xml)
        hoop=donor.find("./worldbody/body[@name='basket_target']")
        if hoop is None:raise RuntimeError('Basket scene does not contain a hoop')
        world=root.find('./worldbody');old=world.find("body[@name='basket_target']")
        if old is not None:world.remove(old)
        world.append(copy.deepcopy(hoop))
        return ET.tostring(root,encoding='unicode')

    def reset(self,*,seed=None,options=None):
        self.chain_phase='catch';self.object_motion='throw';self.env_time=self.catch_timeout
        for k,v in self._catch_flags.items():setattr(self,k,v)
        self.hold_detector.elapsed=0.;self.max_hold_seconds=0.;self.transition=None
        obs,info=super().reset(seed=seed,options=options)
        from gym_dcmm.utils.basket_mobile import sample_target
        self.basket_center=sample_target(self.basket_cfg,self.np_random)
        self.Dcmm.model.body_pos[self.Dcmm.model.body('basket_target').id]=self.basket_center
        self._nominal_front_distance=float(self.basket_cfg.basket_center[1]-self.Dcmm.data.body('arm_base').xpos[1])
        mujoco.mj_forward(self.Dcmm.model,self.Dcmm.data)
        self.chain_start_time=float(self.Dcmm.data.time)
        return self._get_obs(),info

    def hold_measurements(self):
        model,data=self.Dcmm.model,self.Dcmm.data
        ball=model.body('object');palm=model.body('link6')
        bv=np.zeros(6);hv=np.zeros(6)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,ball.id,bv,0)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,palm.id,hv,0)
        displacement=data.body('object').xpos-data.body('link6').xpos
        hand_at_ball=hv[3:]+np.cross(hv[:3],displacement)
        return dict(touching=hand_contact(self),relative_speed=float(np.linalg.norm(bv[3:]-hand_at_ball)),
                    palm_distance=float(np.linalg.norm(displacement)))

    def switch_to_basket(self):
        if self.chain_phase!='catch':raise RuntimeError('Handoff may occur only once')
        before=physical_snapshot(self);model_id=id(self.Dcmm.model);data_id=id(self.Dcmm.data)
        data=self.Dcmm.data
        self.object_motion='throw_basket';self.chain_phase='basket'
        for k,v in self._basket_flags.items():setattr(self,k,v)
        self.start_time=float(data.time) # New TASK clock only; MuJoCo time stays continuous.
        self.basket_rehearsal_episode=False
        self.basket_spawn_check={"source":"continuous_catch","repositioned":False}
        self.terminated=False;self.terminated_reason=None;self.object_throw=False
        self.Dcmm.target_base_vel[:]=0.
        self.Dcmm.target_arm_qpos[:]=data.qpos[15:21]
        self.Dcmm.target_hand_qpos[:]=data.qpos[21:37]
        self.Dcmm.data_arm.qpos[:6]=data.qpos[15:21]
        mujoco.mj_fwdPosition(self.Dcmm.model_arm,self.Dcmm.data_arm)
        self.basket_preparation=dict(local=palm_coordinates(data,data.body('object').xpos).copy(),seconds=0.,stable_seconds=0.)
        self._initialize_basket_state()
        self.basket_cfg.basket_base_front_dist=self._nominal_front_distance
        self.basket_previous_distance=self._basket_parking_state()['distance']
        self.basket_mobile_state['initial_distance']=self.basket_previous_distance
        self.basket_mobile_state['previous_distance']=self.basket_previous_distance
        self.basket_mobile_state['touching']=hand_contact(self)
        self.basket_had_hand_contact=self.basket_mobile_state['touching']
        # Flush queued targets and PID accumulated errors without changing gains or physical state.
        for name,target in [('base',self.Dcmm.target_base_vel),('arm',self.Dcmm.target_arm_qpos),('hand',self.Dcmm.target_hand_qpos)]:
            self.action_buffer[name].clear();self.action_buffer[name].append(target.copy())
        for name in ('arm_pid','hand_pid'):
            pid=getattr(self.Dcmm,name);pid.integral[:]=0.;pid.e_prev[:]=0.;pid.init=False
        self.prev_ee_pos3d=self._get_relative_ee_pos3d().copy()
        self.prev_obj_pos3d=self._get_relative_object_pos3d().copy()
        self.info=self._get_info()
        after=physical_snapshot(self)
        changed=[k for k in before if not np.array_equal(before[k],after[k])]
        if changed or id(self.Dcmm.model)!=model_id or id(self.Dcmm.data)!=data_id:
            raise RuntimeError('Physical state changed during handoff: '+str(changed))
        self.transition=dict(sim_time=float(data.time),catch_elapsed=float(data.time-self.chain_start_time),
                             physics_before=snapshot_hash(before),physics_after=snapshot_hash(after),
                             model_identity_preserved=True,physical_state_preserved=True,
                             ball_local=self.basket_preparation['local'].tolist(),
                             arm_qpos=data.qpos[15:21].tolist(),hand_qpos=data.qpos[21:37].tolist())
        return self._get_obs()

    def step(self,action):
        was_catch=self.chain_phase=='catch'
        obs,reward,terminated,truncated,info=super().step(action)
        info=dict(info);info.pop('imgs',None)
        if was_catch:
            measure=self.hold_measurements()
            stable=self.hold_detector.update(self.steps_per_policy*self.Dcmm.model.opt.timestep,
                measure['touching'],measure['relative_speed'],measure['palm_distance'],not terminated)
            self.max_hold_seconds=max(self.max_hold_seconds,self.hold_detector.elapsed)
            info.update(chain_hold=measure,chain_hold_seconds=self.hold_detector.elapsed)
            if stable and not terminated:
                obs=self.switch_to_basket();terminated=False;truncated=False
                info.update(chain_switched=True,chain_phase='basket',success=False)
            else:
                # Throw's timeout success is not a valid chained stable-catch success.
                info['chain_switched']=False
                if terminated or truncated:
                    info['chain_failure']='catch_'+str(self.terminated_reason or ('terminated_without_reason' if terminated else 'timeout_without_stable_hold'))
        else:
            info['chain_switched']=False
            if terminated or truncated:info['chain_failure']=None if info.get('safe_basket_success') else 'basket_'+str(self.terminated_reason or 'timeout')
        info['chain_phase']=self.chain_phase
        return obs,reward,terminated,truncated,info

    def _initialize_basket_state(self):
        """Logical state block copied from DcmmVecEnv.reset; never steps or resets physics."""
        if self.object_motion == "throw_basket":
            self.basket_settle_steps = 0
            self.basket_phase = 'preparing'
            self.basket_settled_time = 0.0
            self.basket_had_hand_contact = False
            self.basket_previous_velocity = None
            self.basket_release_pending = 0.
            self.basket_no_contact_seconds = 0.
            self.basket_contact_total = 0.
            self.basket_contact_run = 0.
            self.basket_last_contact = None
            self.basket_release_snapshot = None
            self.basket_release_count = 0
            self.basket_regrasp_count = 0
            self.basket_hand_clearance = 0.
            self.basket_release_quality = 0.
            self.basket_throw_valid = False
            self.basket_release_position = self.Dcmm.data.qpos[37:40].copy()
            self.basket_release_time = float(self.Dcmm.data.time)
            self.basket_support_time = 0.
            self.basket_reward_state = {}
            self._basket_arm_diagnostics = None
            self.basket_aim_hit_time = None
            self.basket_simple_state = {}
            if self.basket_fixed_physical:
                self.basket_simple_state['arm_drive'] = dict(arm_drive_origin=(
                    self.Dcmm.data.body('link6').xpos-self.Dcmm.data.body('arm_base').xpos).copy())
            if self.basket_carry_reward:
                from gym_dcmm.utils.basket_carry import new_state
                self.basket_simple_state['carry'] = new_state(
                    self.Dcmm.data.body('link6').xpos-self.Dcmm.data.body('arm_base').xpos,
                    self.basket_preparation['local'])
            if self.basket_finger_reward:
                from gym_dcmm.utils.basket_fingers import HAND_INDICES, new_state as finger_state
                joint_ids = [int(np.flatnonzero(self.Dcmm.model.jnt_qposadr == 21+i)[0])
                             for i in HAND_INDICES]
                self.basket_finger_state = finger_state(self.Dcmm.data.qpos[21:37][HAND_INDICES],
                    self.Dcmm.model.jnt_range[joint_ids], self.basket_finger_options)
            self.basket_arm_self_collision = False
            self.basket_self_contact_details = []
            self.basket_robot_collision = False
            self.basket_collision_details = []
            self.basket_acceleration_pending = 0.
            self.basket_retention_pending = 0.
            self.basket_downward_pending = 0.
            self.basket_slip_speed = 0.
            self.basket_release_candidate = None
            self.basket_valid_throw_confirmed = False
            self.basket_throw_reward_paid = False
            self.basket_throw_pending = 0.
            self.basket_throw_check = dict(reason='no_candidate')
            self.basket_reward_totals = {}
            self.basket_arm_attempts = 0
            self.basket_arm_ik_successes = 0
            self.basket_previous_flight_distance = None
            self.basket_initial_arm_base_position = self.Dcmm.data.body('arm_base').xpos.copy()
            if self.mobile_basket:
                # Preserve the fixed policy's target-to-arm_base offset, independent
                # of the new target and of the robot's subsequent movement.
                self.basket_cfg.basket_base_front_dist = float(
                    self.basket_cfg.basket_center[1]-self.basket_initial_arm_base_position[1])
            self.basket_previous_distance = self._basket_parking_state()['distance']
            if self.mobile_basket:
                from gym_dcmm.utils.basket_mobile import new_state
                self.basket_mobile_state = new_state(self.basket_previous_distance)
                if self.basket_rehearsal_episode:
                    # A separate fixed-target rehearsal episode starts in the
                    # original reset state. Never teleport/regrasp after driving.
                    low, high = self.basket_cfg.mobile_hold_time_range
                    delay = float(self.np_random.uniform(low,high)) if high > 0 else 0.
                    if delay > 0:
                        self.basket_mobile_state['hold_until'] = delay
                    else:
                        self.basket_mobile_state.update(unlocked=True, ready_time=0.,
                            settle_seconds=self.basket_cfg.mobile_settle_seconds)
                self.basket_carry_arm = self.Dcmm.target_arm_qpos.copy()
                self.basket_carry_hand = self.Dcmm.target_hand_qpos.copy()
