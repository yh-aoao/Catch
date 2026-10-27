"""Frozen fixed-base throwing skill; PPO optimizes a separate navigation actor."""
import hashlib
from pathlib import Path

import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd


class FrozenThrowTeacher:
    observation_version = 'fixed35_parking02_v1'

    def __init__(self, checkpoint, device='cpu'):
        if (checkpoint.get('action_layout') != 'arm6_hand12' or
                checkpoint.get('basket_arm_control') != 'joint_velocity'):
            raise ValueError('Throw teacher must be an arm6_hand12 joint_velocity fixed-base checkpoint')
        state = checkpoint['model']
        first = state['actor_mlp.mlp.0.weight']
        if first.shape[1] != 35 or state['mu.weight'].shape[0] != 18:
            raise ValueError('Throw teacher requires 35 observations and 18 actions')
        normalization = checkpoint.get('running_mean_std')
        if normalization is None or normalization['running_mean'].shape != (35,):
            raise ValueError('Throw teacher requires its original 35-dimensional normalization')
        units = [value.shape[0] for key, value in state.items()
                 if key.startswith('actor_mlp.') and key.endswith('.weight')]
        self.model = ActorCritic(dict(actor_units=units, actions_num=18,
            input_shape=(35,), separate_value_mlp=any(k.startswith('value_mlp.') for k in state)))
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.normalizer = RunningMeanStd((35,)).to(device).eval()
        self.normalizer.load_state_dict(normalization, strict=True)
        self.normalizer.requires_grad_(False)
        self.source_sha256 = checkpoint.get('source_sha256', '')
        self.source_name = checkpoint.get('source_name', '')

    @classmethod
    def from_file(cls, filename, device='cpu'):
        checkpoint = torch.load(filename, map_location=device)
        teacher = cls(checkpoint, device)
        teacher.source_sha256 = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        teacher.source_name = Path(filename).name
        return teacher

    @torch.no_grad()
    def actions(self, raw_observation):
        if raw_observation.shape[-1] not in (35, 51):
            raise ValueError('Throw teacher expects raw fixed35 or mobile51 observations')
        obs = raw_observation[:, :35].clone()
        # Fixed-base training used initial arm_base_y + .2 for its unused parking
        # target. Preserve these two legacy features; real hoop/ball state stays
        # measured. Feeding the mobile near-zero parking error changes the policy.
        obs[:, -2] = 0.
        obs[:, -1] = .2
        return self.model.act_inference(dict(obs=self.normalizer(obs))).clamp(-1., 1.)

    def export(self):
        return dict(model=self.model.state_dict(), running_mean_std=self.normalizer.state_dict(),
            action_layout='arm6_hand12', basket_arm_control='joint_velocity',
            source_sha256=self.source_sha256, source_name=self.source_name,
            teacher_observation_version=self.observation_version)
