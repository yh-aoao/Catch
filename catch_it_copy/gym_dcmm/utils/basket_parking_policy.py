"""The stage-one parking actor, frozen and embedded in stage-two checkpoints."""
import torch

from gym_dcmm.algs.ppo_dcmm.models_track import ActorCritic
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd
from gym_dcmm.utils.basket_mobile import DEFAULTS


class FrozenParkingPolicy:
    def __init__(self, checkpoint, options, device):
        if (checkpoint.get('basket_training_stage') != 'parking' or
                checkpoint.get('action_layout') != 'base2_parking' or
                checkpoint.get('basket_observation_version') != 'mobile51_v1'):
            raise ValueError('Stage two requires a basket_stage1 parking checkpoint (51 observations, base2 actions)')
        source_options = dict(DEFAULTS, **checkpoint['basket_mobile_options'])
        target_options = dict(DEFAULTS, **options)
        # These options supervise stage transition/rehearsal in the environment;
        # they do not change the frozen actor's action scaling or target geometry.
        for key in ('parking_only', 'hold_position', 'settle_contact_grace',
                    'release_speed_tolerance', 'release_position_tolerance',
                    'post_ready_seconds', 'fixed_throw_probability', 'hold_only',
                    'hold_time_range', 'collision_free_spawn', 'hold_aim_bonus', 'hold_ball_speed_tolerance'):
            source_options.pop(key, None)
            target_options.pop(key, None)
        if source_options != target_options:
            raise ValueError('Parking and throwing stages must use matching geometry/control options')
        state = checkpoint['model']
        units = [value.shape[0] for key, value in state.items()
                 if key.startswith('actor_mlp.') and key.endswith('.weight')]
        self.model = ActorCritic(dict(actor_units=units, actions_num=2, input_shape=(51,),
            separate_value_mlp=any(k.startswith('value_mlp.') for k in state)))
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.normalizer = RunningMeanStd((51,)).to(device).eval()
        self.normalizer.load_state_dict(checkpoint['running_mean_std'], strict=True)
        self.normalizer.requires_grad_(False)
        self.options = checkpoint['basket_mobile_options']

    @torch.no_grad()
    def actions(self, raw_observation):
        result = self.model.act_inference(dict(obs=self.normalizer(raw_observation))).clamp(-1., 1.)
        # Inference alone must not issue a new drive command after settling.
        return result.masked_fill(raw_observation[:, 47:48] > .5, 0.)

    def export(self):
        return dict(model=self.model.state_dict(), running_mean_std=self.normalizer.state_dict(),
            basket_training_stage='parking', action_layout='base2_parking',
            basket_observation_version='mobile51_v1', basket_mobile_options=self.options)
