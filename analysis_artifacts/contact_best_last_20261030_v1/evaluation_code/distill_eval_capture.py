"""Read-only reset-state capture used only by independent student evaluation."""
import hashlib
import json
import sys

import numpy as np


def make_env(**kwargs):
    from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
    env = DcmmVecEnv(**kwargs)
    original = env.reset

    def reset(*args, **kw):
        obs, info = original(*args, **kw)
        model, data = env.Dcmm.model, env.Dcmm.data
        if 'model_experiments.run' in sys.modules:
            raise RuntimeError('Unavailable historical training launcher was unexpectedly imported')
        digest = hashlib.sha256()
        for owner, names in [(data, ('qpos', 'qvel')),
                             (model, ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                                      'body_pos', 'geom_size', 'geom_friction', 'geom_solref',
                                      'geom_solimp', 'geom_condim', 'geom_contype', 'geom_conaffinity',
                                      'dof_damping', 'dof_armature', 'cam_pos', 'cam_quat'))]:
            for name in names:
                array = np.asarray(getattr(owner, name))
                digest.update(name.encode())
                digest.update(array.tobytes())
        extras = {}
        for name in ('k_arm', 'k_drive', 'k_hand', 'k_obs_arm', 'k_obs_object', 'k_obs_base', 'k_obs_hand'):
            if hasattr(env, name):
                extras[name] = np.asarray(getattr(env, name)).tolist()
        extras['control_delay'] = {key: len(value) for key,value in env.action_buffer.items()}
        extras['timestep'] = float(model.opt.timestep)
        extras['gravity'] = model.opt.gravity.tolist()
        extras['integrator'] = int(model.opt.integrator)
        extras['bounce_substeps'] = int(getattr(env, '_bounce_physics_substeps', 1))
        if extras['bounce_substeps'] != 1:
            raise RuntimeError('Expected original legacy physics substeps')
        digest.update(json.dumps(extras, sort_keys=True).encode())
        bid = model.body('object').id
        jid = model.body_jntadr[bid]
        q, v = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
        initial = dict(physical_hash=digest.hexdigest(), extras=extras,
                       qpos=data.qpos.tolist(), qvel=data.qvel.tolist(),
                       object_position=data.qpos[q:q+7].tolist(),
                       object_velocity=data.qvel[v:v+6].tolist(),
                       object_mass=float(model.body_mass[bid]),
                       object_inertia=model.body_inertia[bid].tolist(),
                       radius=float(model.geom_size[env.object_id, 0]))
        obs['_evaluation_initial_json'] = json.dumps(initial, sort_keys=True)
        return obs, info
    env.reset = reset
    return env
