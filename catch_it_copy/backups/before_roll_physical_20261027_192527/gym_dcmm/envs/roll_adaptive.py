"""Reward-only opt-in wrapper around the existing Roll Track/Catch environments."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid

import gymnasium as gym
import mujoco
import numpy as np

from gym_dcmm.utils.roll_adaptive_wait import AdaptiveWaitConfig, interception, terms
from gym_dcmm.utils.roll_rewards import hand_collision_ids

VERSION = 'adaptive_wait_v1'
BASELINE = 'backups/roll_before_adaptive_wait_20261027_001703/manifest.json'


class AdaptiveRollReward(gym.Wrapper):
    # Gymnasium 0.29 make_vec assigns spec directly to a factory's returned env.
    # The registered factory already recreates this wrapper, so expose its base
    # spec rather than adding a second, separately reconstructed wrapper spec.
    @property
    def spec(self):
        return self.env.spec

    @spec.setter
    def spec(self, value):
        self.env.spec = value

    def __init__(self, env, options=None):
        super().__init__(env)
        self.cfg = AdaptiveWaitConfig(**(options or {}))
        self.previous_palm = None
        self.was_waiting = None
        self.left_table = False
        self.sums = {}
        self.steps = 0
        self.episode = 0
        self.log_file = None
        self.log_token = uuid.uuid4().hex[:12]

    def set_roll_reward_log_dir(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.log_file = directory / f'episodes_{os.getpid()}_{self.log_token}.jsonl'

    def reset(self, **kwargs):
        self.previous_palm = None
        self.was_waiting = None
        self.left_table = False
        self.sums = {}
        self.steps = 0
        self.episode += 1
        obs, info = self.env.reset(**kwargs)
        info['roll_reward_version'] = VERSION
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env = self.env.unwrapped
        model, data = env.Dcmm.model, env.Dcmm.data
        bid = model.body('object').id
        position = data.body('object').xpos.copy()
        radius = float(model.geom_size[env.object_id, 0])
        table = model.geom('table_surface').id
        edge = float(data.geom_xpos[table, 1] - model.geom_size[table, 1])
        top = float(data.geom_xpos[table, 2] + model.geom_size[table, 2])
        hand_ids = hand_collision_ids(model, env.hand_start_id)
        object_contacts = env.contacts['object_contacts']
        table_contact = bool(np.any(np.asarray(object_contacts) == table))
        hand_contact = bool(np.any(np.isin(object_contacts, hand_ids)))
        if position[1]+radius < edge and not table_contact:
            self.left_table = True
        waiting = not self.left_table
        palm = (data.geom_xpos[env.hand_start_id].copy()
                + data.body('link6').xmat.reshape(3, 3)[:, 1]*radius)
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, position, bid)
        velocity = jac @ data.qvel
        mujoco.mj_jac(model, data, jac, None, palm, int(model.geom_bodyid[env.hand_start_id]))
        palm_velocity = jac @ data.qvel
        if waiting:
            goal, prediction = interception(position, velocity, radius, edge, top,
                                             model.opt.gravity[2], self.cfg)
        else:
            goal, prediction = position.copy(), {}
        hand_front = float(np.max(data.geom_xpos[hand_ids, 1]+model.geom_rbound[hand_ids]))
        previous = self.previous_palm if waiting == self.was_waiting else None
        new_terms = terms(palm, previous, palm_velocity, goal, waiting, hand_front,
                          edge, hand_contact, table_contact, self.cfg)
        old_terms = info['roll_wait_terms']
        removed = float(sum(old_terms.values()))
        added = float(sum(new_terms.values()))
        result = float(reward) - removed + added
        self.previous_palm = palm.copy()
        self.was_waiting = waiting
        self.steps += 1
        for key, value in dict(new_terms, old_reward=float(reward), removed_wait=removed,
                               added_wait=added, reward=result, waiting=float(waiting)).items():
            self.sums[key] = self.sums.get(key, 0.)+float(value)
        info['roll_reward_version'] = VERSION
        info['roll_legacy_wait_terms'] = dict(old_terms)
        info['roll_wait_terms'] = new_terms
        info['roll_wait_target'] = goal
        info['roll_waiting'] = waiting
        info['roll_adaptive_wait'] = dict(prediction, hand_front_y=hand_front, edge_y=edge,
                                           palm_z=float(palm[2]), target_z=float(goal[2]),
                                           speed=float(np.linalg.norm(velocity[:2])),
                                           base_reward=float(reward), removed=removed,
                                           added=added, waiting=waiting, terms=new_terms)
        if terminated or truncated:
            summary = dict(version=VERSION, episode=self.episode, steps=self.steps,
                           reason=str(getattr(env, 'terminated_reason', None) or 'timeout'),
                           success=bool(info.get('roll_eval_success', info.get('success', False))),
                           terminated=bool(terminated), truncated=bool(truncated),
                           sums=self.sums.copy(), config=asdict(self.cfg))
            info['roll_adaptive_episode'] = summary
            if self.log_file:
                with self.log_file.open('a') as file:
                    file.write(json.dumps(summary)+'\n')
        return obs, result, terminated, truncated, info


def make_roll_adaptive(roll_adaptive_wait=None, test_object='default', **kwargs):
    if kwargs.get('object_motion') != 'roll':
        raise ValueError('Adaptive Roll reward requires object_motion=roll')
    if kwargs.get('roll_wait_guard', False):
        raise ValueError('Adaptive reward must be evaluated without the hard waiting guard')
    if test_object != 'default':
        from gym_dcmm.envs.object_test import make_object_test
        env = make_object_test(test_object, **kwargs)
    else:
        from gym_dcmm.envs.roll_compat import make_roll_645edc4
        env = make_roll_645edc4(kwargs)
    return AdaptiveRollReward(env, roll_adaptive_wait)


def write_provenance(output, config):
    root = Path(__file__).resolve().parents[2]
    paths = ['train_DCMM.py', 'configs/config.yaml', 'configs/train/DcmmPPO.yaml',
             'gym_dcmm/__init__.py', 'gym_dcmm/envs/roll_adaptive.py',
             'gym_dcmm/utils/roll_adaptive_wait.py', 'gym_dcmm/utils/roll_waiting.py',
             'gym_dcmm/envs/roll_compat.py', 'gym_dcmm/envs/DcmmVecEnv.py',
             'gym_dcmm/envs/DcmmVecEnv_roll_645edc4.py',
             'configs/env/DcmmCfg_roll_645edc4.py', 'configs/env/DcmmCfg_roll_track_07c176f.py']
    paths += [str(p.relative_to(root)) for p in (root/'gym_dcmm/algs/ppo_dcmm_645edc4').glob('*.py')]
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
    checkpoints = {}
    for key in ('checkpoint_tracking', 'checkpoint_catching'):
        name = config.get(key)
        if name:
            path = Path(name)
            checkpoints[key] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    baseline = root/BASELINE
    result = dict(version=VERSION, baseline_manifest=BASELINE,
                  baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                  sources=sources, checkpoints=checkpoints,
                  implementation='base reward minus old four wait terms plus adaptive wait terms',
                  hard_wait_guard=False)
    (Path(output)/'roll_reward_provenance.json').write_text(json.dumps(result, indent=2))
