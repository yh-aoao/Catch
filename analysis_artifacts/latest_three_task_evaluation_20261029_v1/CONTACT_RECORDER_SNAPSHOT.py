"""Paired teacher tests and substep floor-contact diagnostics for Roll/Bounce."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import mujoco
import numpy as np
import torch


class ContactRecorder:
    def __init__(self, env):
        from gym_dcmm.utils.roll_rewards import hand_collision_ids
        self.env = env
        self.model, self.data = env.Dcmm.model, env.Dcmm.data
        model = self.model
        body = model.body('object').id
        joint = model.body_jntadr[body]
        self.q, self.v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        self.ids = set(np.flatnonzero(model.geom_bodyid == body).tolist())
        self.hands = set(hand_collision_ids(model, env.hand_start_id).tolist())
        self.floor = model.geom('floor').id
        self.last = self.state()
        self.floor_contact = False
        self.impacts = []
        self.had_hand = False
        self.hand_duration = 0.
        self.max_hand_duration = 0.
        self.max_speed = 0.
        self.max_height = float(self.last['position'][2])

    def state(self):
        return dict(time=float(self.data.time), position=self.data.qpos[self.q:self.q+3].tolist(),
                    velocity=self.data.qvel[self.v:self.v+6].tolist())

    def update(self):
        floor, hand = False, False
        contact = None
        for index, c in enumerate(self.data.contact):
            a, b = int(c.geom1), int(c.geom2)
            if (a in self.ids) == (b in self.ids):
                continue
            other = b if a in self.ids else a
            if other == self.floor:
                floor, contact = True, c
            hand |= other in self.hands
        current = self.state()
        self.max_speed = max(self.max_speed, float(np.linalg.norm(current['velocity'][:3])))
        self.max_height = max(self.max_height, current['position'][2])
        if floor and not self.floor_contact:
            self.impacts.append(dict(before=self.last, first_contact=current,
                                     after=None, before_any_hand_contact=not self.had_hand,
                                     contact_dim=int(contact.dim), friction=contact.friction.tolist(),
                                     solref=contact.solref.tolist(), solimp=contact.solimp.tolist()))
        elif not floor and self.floor_contact:
            self.impacts[-1]['after'] = current
        dt = max(0., current['time'] - self.last['time'])
        self.hand_duration = self.hand_duration+dt if hand else 0.
        self.max_hand_duration = max(self.max_hand_duration, self.hand_duration)
        self.had_hand |= hand
        self.floor_contact, self.last = floor, current


def make_env(motion, name, collision='auto', env_time=2.5):
    kwargs = dict(task='Catching', object_motion=motion, object_eval=False,
                  viewer=False, render_mode=None, env_time=env_time, steps_per_policy=20)
    if name != 'default':
        from gym_dcmm.envs.object_test import make_object_test
        return make_object_test('ycb:'+name, ycb_collision=collision, **kwargs)
    if motion == 'roll':
        from gym_dcmm.envs.roll_compat import make_roll_645edc4
        return make_roll_645edc4(kwargs)
    from gym_dcmm.envs.DcmmVecEnv import BounceEnv
    return BounceEnv(**kwargs)


def physical_parameters(env):
    model = env.Dcmm.model
    body = model.body('object').id
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    ids = np.flatnonzero((model.geom_bodyid == body) &
                        ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    return dict(mass=float(model.body_mass[body]), radius=float(model.geom_size[env.object_id, 0]),
                inertia=model.body_inertia[body].tolist(), com=model.body_ipos[body].tolist(),
                inertia_quat=model.body_iquat[body].tolist(),
                armature=model.dof_armature[dof:dof+6].tolist(), damping=model.dof_damping[dof:dof+6].tolist(),
                collision_geoms=[dict(name=model.geom(int(g)).name, type=int(model.geom_type[g]),
                                      solref=model.geom_solref[g].tolist(),
                                      friction=model.geom_friction[g].tolist(), condim=int(model.geom_condim[g]))
                                 for g in ids])


def match_default_physics(env, reference):
    """Diagnostic only: preserve YCB appearance but copy the original ball physics."""
    from gym_dcmm.envs.object_test import set_constants_preserving_state
    model, source = env.Dcmm.model, reference.Dcmm.model
    body, source_body = model.body('object').id, source.body('object').id
    for key in ('geom_size', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
        value = getattr(source, key)[reference.object_id if key.startswith('geom_') else source_body].copy()
        env._test_object_constants[key] = value
        getattr(model, key)[env.object_id if key.startswith('geom_') else body] = value
    for g in np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError('Physics matching requires sphere collisions')
        model.geom_size[g] = source.geom_size[reference.object_id]
        model.geom_rbound[g] = source.geom_size[reference.object_id, 0]
        if hasattr(model, 'geom_aabb'):
            model.geom_aabb[g] = [0., 0., 0., *([source.geom_size[reference.object_id, 0]]*3)]
    dof, source_dof = model.jnt_dofadr[model.body_jntadr[body]], source.jnt_dofadr[source.body_jntadr[source_body]]
    model.dof_armature[dof:dof+6] = source.dof_armature[source_dof:source_dof+6]
    model.dof_damping[dof:dof+6] = source.dof_damping[source_dof:source_dof+6]
    set_constants_preserving_state(model, env.Dcmm.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions', nargs='+', choices=['roll', 'bounce'], default=['roll', 'bounce'])
    parser.add_argument('--objects', nargs='+', choices=['default', 'tennis_ball', 'baseball'],
                        default=['default', 'tennis_ball', 'baseball'])
    parser.add_argument('--collision', choices=['auto', 'mesh', 'sphere'], default='auto')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env-time', type=float, default=2.5)
    parser.add_argument('--match-default-physics', action='store_true',
                        help='DIAGNOSTIC: replace YCB mass/radius/inertia/COM/armature with original ball values')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.env_time <= 0:
        parser.error('Episodes and duration must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    from distillation.core import observation_array, ActionAdapter
    from distillation.teachers import ROOT, Teacher
    config = json.loads((ROOT/'distillation/configs/three_tasks.json').read_text())
    torch.set_num_threads(1)
    summary = {}
    for motion in args.motions:
        teacher = Teacher(config['tasks'][motion], 'cpu')
        adapter = ActionAdapter(config['tasks'][motion])
        for name in args.objects:
            key = motion+'_'+name
            np.random.seed(args.seed)
            random.seed(args.seed)
            env = make_env(motion, name, args.collision, args.env_time)
            if args.match_default_physics and name != 'default':
                np.random.seed(args.seed)
                random.seed(args.seed)
                reference = make_env(motion, 'default', args.collision, args.env_time)
                try:
                    match_default_physics(env, reference)
                finally:
                    reference.close()
            original_contacts = env._get_contacts
            recorder = None

            def record_contacts():
                result = original_contacts()
                if recorder is not None:
                    recorder.update()
                return result

            env._get_contacts = record_contacts
            rows = []
            try:
                for seed in range(args.seed, args.seed+args.episodes):
                    recorder = None
                    np.random.seed(seed)
                    random.seed(seed)
                    obs, _ = env.reset()
                    recorder = ContactRecorder(env)
                    row = dict(seed=seed, initial_position=env.object_pos3d.tolist(),
                               initial_velocity=env.object_vel6d.tolist(),
                               actual_reset_state=recorder.state(),
                               actual_reset_arm_qpos=env.Dcmm.data.qpos[15:21].tolist(),
                               physics=physical_parameters(env), trace=[])
                    teacher.parity_check(observation_array(obs)[None])
                    for _ in range(int(args.env_time/.04)+5):
                        command = teacher.predict(observation_array(obs)[None])[0]
                        obs, _, terminated, truncated, info = env.step(adapter.action_dict(command))
                        row['trace'].append(recorder.state())
                        if terminated or truncated:
                            break
                    else:
                        raise RuntimeError('Episode failed to end within expected steps')
                    row.update(success=bool(info.get('roll_eval_success', False) if motion == 'roll' else info['success']),
                               environment_success=bool(info.get('success', False)),
                               reason=str(info.get('terminated_reason') or env.terminated_reason or 'timeout'),
                               hand_contact=recorder.had_hand, max_contiguous_hand_contact_s=recorder.max_hand_duration,
                               max_speed=recorder.max_speed, max_height=recorder.max_height,
                               floor_impacts=recorder.impacts, elapsed=float(info['env_time']))
                    rows.append(row)
                    with (args.output/(key+'.jsonl')).open('a') as file:
                        file.write(json.dumps(row)+'\n')
                    print(key, seed, 'success=', row['success'], 'hand=', row['hand_contact'],
                          'max_v=', round(row['max_speed'], 3), 'reason=', row['reason'], flush=True)
            finally:
                env.close()
            summary[key] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                                hand_contact_episodes=sum(r['hand_contact'] for r in rows),
                                max_speed=max(r['max_speed'] for r in rows),
                                max_height=max(r['max_height'] for r in rows),
                                reasons=dict(Counter(r['reason'] for r in rows)),
                                teacher_sha256=teacher.hash, collision=args.collision)
            summary[key]['physics_matched_to_original_ball'] = args.match_default_physics
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
