"""Frozen fixed35/arm18 Basket skill embedded in a learned mobile policy."""
import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd


class FrozenThrowPolicy:
    def __init__(self, checkpoint, finger_options, carry_options, device):
        if (checkpoint.get('action_layout') != 'arm6_hand12'
                or checkpoint.get('basket_arm_control') != 'joint_velocity'
                or checkpoint.get('basket_fixed_protocol') != 'aim_carry_fingers_v1'
                or checkpoint.get('basket_finger_options') != finger_options
                or checkpoint.get('basket_carry_options') != carry_options):
            raise ValueError('Mobile41 requires matching fixed35 finger/carry joint-velocity throw checkpoint')
        state = checkpoint['model']
        if (state['actor_mlp.mlp.0.weight'].shape[1] != 35
                or state['mu.weight'].shape[0] != 18):
            raise ValueError('Frozen throw requires exactly 35 observations and 18 actions')
        units = [v.shape[0] for k, v in state.items()
                 if k.startswith('actor_mlp.') and k.endswith('.weight')]
        self.model = ActorCritic(dict(actor_units=units, actions_num=18,
            input_shape=(35,), separate_value_mlp=any(k.startswith('value_mlp.') for k in state)))
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.normalizer = RunningMeanStd((35,)).to(device).eval()
        self.normalizer.load_state_dict(checkpoint['running_mean_std'], strict=True)
        self.normalizer.requires_grad_(False)
        self.finger_options = dict(finger_options)
        self.carry_options = dict(carry_options)

    @torch.no_grad()
    def actions(self, raw_observation):
        if raw_observation.ndim != 2 or raw_observation.shape[1] != 51:
            raise ValueError('Frozen throw expects unnormalized mobile51 observations')
        result = raw_observation.new_zeros((len(raw_observation), 18))
        active = raw_observation[:, 47] > .5
        if active.any():
            fixed = raw_observation[active, :35].clone()
            # Fixed35's parking Y feature used initial-Y + 0.2, while mobile51
            # reports error to the calibrated throw stand. At that stand it is 0.
            fixed[:, 34] += .2
            result[active] = self.model.act_inference(dict(obs=self.normalizer(fixed))).clamp(-1., 1.)
        return result

    def export(self):
        return dict(model=self.model.state_dict(), running_mean_std=self.normalizer.state_dict(),
            action_layout='arm6_hand12', basket_arm_control='joint_velocity',
            basket_fixed_protocol='aim_carry_fingers_v1',
            basket_finger_options=self.finger_options, basket_carry_options=self.carry_options)
