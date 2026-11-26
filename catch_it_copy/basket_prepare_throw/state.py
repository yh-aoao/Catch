"""Reset-only replay of genuine preparation states, including servo and delay history."""
import copy,hashlib
import mujoco
import numpy as np
VERSION='physical_ready_state_v1'
ENV_FIELDS=('last_action','prev_ee_pos3d','prev_obj_pos3d','_low_speed_time','_last_touch',
    '_no_hand_seconds','reset_stats','_initial_base_xy','action_buffer','basket_mobile_state','k_act','object_throw')
DCMM_FIELDS=('target_base_vel','target_arm_qpos','target_hand_qpos','steer_ang','drive_vel')
def fingerprint(model):
    digest=hashlib.sha256()
    for name in ('geom_size','geom_friction','geom_pos','body_mass','body_inertia','jnt_range','dof_damping'):
        digest.update(np.asarray(getattr(model,name)).tobytes())
    digest.update(str((model.nq,model.nv,model.nu,model.opt.timestep)).encode())
    return digest.hexdigest()
def capture(env):
    m,d=env.Dcmm.model,env.Dcmm.data
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec)
    return dict(version=VERSION,fingerprint=fingerprint(m),physics=state,body_pos=m.body_pos.copy(),
        target=env.basket_center.copy(),env={k:copy.deepcopy(getattr(env,k)) for k in ENV_FIELDS},
        dcmm={k:copy.deepcopy(getattr(env.Dcmm,k)) for k in DCMM_FIELDS if hasattr(env.Dcmm,k)},
        pid={k:copy.deepcopy(getattr(env.Dcmm,k).__dict__) for k in ('steer_pid','drive_pid','arm_pid','hand_pid')},
        arm_qpos=env.Dcmm.data_arm.qpos.copy(),arm_qvel=env.Dcmm.data_arm.qvel.copy(),
        observation=env.observation().copy())
def restore(env,snapshot):
    m,d=env.Dcmm.model,env.Dcmm.data
    if snapshot['version']!=VERSION or snapshot['fingerprint']!=fingerprint(m):
        raise ValueError('Ready bank physics differs from current environment')
    m.body_pos[:]=snapshot['body_pos'];env.basket_center[:]=snapshot['target']
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(m,d,snapshot['physics'],spec);mujoco.mj_forward(m,d)
    # mj_forward may modify warm-start values: retain exact saved integration state.
    mujoco.mj_setState(m,d,snapshot['physics'],spec)
    for k,v in snapshot['env'].items():setattr(env,k,copy.deepcopy(v))
    for k,v in snapshot['dcmm'].items():setattr(env.Dcmm,k,copy.deepcopy(v))
    for k,v in snapshot['pid'].items():getattr(env.Dcmm,k).__dict__.update(copy.deepcopy(v))
    env.Dcmm.data_arm.qpos[:]=snapshot['arm_qpos'];env.Dcmm.data_arm.qvel[:]=snapshot['arm_qvel']
    mujoco.mj_fwdPosition(env.Dcmm.model_arm,env.Dcmm.data_arm)
    # Only task accounting restarts; physical/controller state is never zeroed or reseated.
    env.steps=0;env.start_time=float(d.time);env.terminated=False;env.terminated_reason=None
    env._scored=False;env._released_once=False;env._release=None;env._release_reward=0.
    env._distance_reward_paid=False;env._release_distance_value=None;env._quality_best=0.
    env._ever_intentional_launch=False;env._carry_slip_integral=0.;env._carry_exposure=0.;env._carry_slip_max=0.
    env.basket_robot_collision=False;env.basket_arm_self_collision=False
    env._navigation_best=env._navigation_distance();env._reward_totals={}
