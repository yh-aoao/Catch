"""Basket two-stage PPO: parking actor first, then arm/hand with frozen parking.

Uses the common PPO rollout/update implementation, but the trainable action
space is base2 in stage one and arm6+hand12 in stage two. Optimizers, input
normalizers and checkpoints of the two actors are independent.
"""
from .ppo_dcmm_catch_one_stage import PPO_Catch_OneStage


class PPO_Basket_TwoStage(PPO_Catch_OneStage):
    def __init__(self, env, output_dif, full_config):
        if full_config.get('basket_training_stage') not in ('parking', 'throw', 'hold'):
            raise ValueError('Basket two-stage PPO requires parking or throw stage')
        super().__init__(env, output_dif, full_config)
