"""Second-stage Basket PPO: both navigation and inherited throwing keep learning."""
import hashlib
import os

import torch

from .ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from .basket_mobile_joint import JointActorCritic, JointNormalizer


class PPO_Basket_MobileJoint(PPO_Catch_OneStage):
    def __init__(self, env, output_dif, full_config):
        if full_config.basket_training_stage != 'mobile_joint':
            raise ValueError('Joint Basket PPO requires mobile_joint stage')
        if not full_config.train.network.get('separate_value_mlp', True):
            raise ValueError('Joint Basket requires a separate critic')
        super().__init__(env,output_dif,full_config)
        self.model = JointActorCritic(list(self.network_config.mlp.units)).to(self.device)
        self.running_mean_std = JointNormalizer().to(self.device)
        self.input_normalization_frozen = False
        if not full_config.test:
            source_path=full_config.basket_throw_checkpoint
            checkpoint=torch.load(source_path,map_location=self.device)
            self.model.import_throw(checkpoint,self.running_mean_std,self.finger_options,self.carry_options)
            with open(source_path,'rb') as stream:digest=hashlib.sha256(stream.read()).hexdigest()
            self.basket_transfer_metadata=dict(version='joint_fixed35_throw_base2_v1',
                source_name=os.path.basename(source_path),source_sha256=digest)
            print('[basket-mobile41-joint] arm18 initialized from '+source_path+'; base2 and arm18 both trainable',flush=True)
        base,throw,critic=self.model.actor_groups()
        self.throw_lr_scale=float(full_config.basket_throw_lr_scale)
        if not 0 < self.throw_lr_scale <= 1:
            raise ValueError('basket_throw_lr_scale must be in (0,1]')
        self.optimizer=torch.optim.Adam([
            dict(params=base,lr=self.init_lr,lr_scale=1.,name='base'),
            dict(params=throw,lr=self.init_lr*self.throw_lr_scale,lr_scale=self.throw_lr_scale,name='throw'),
            dict(params=critic,lr=self.init_lr,lr_scale=1.,name='critic')],eps=1e-5)
        print('[basket-mobile41-joint] PPO actions=20; throw/base learning-rate ratio='+str(self.throw_lr_scale),flush=True)

    def save(self,name):
        weights=dict(model=self.model.state_dict(),running_mean_std=self.running_mean_std.state_dict(),
            value_mean_std=self.value_mean_std.state_dict(),
            joint_actor_version='phase_conditioned_base2_arm18_v1',
            action_layout='joint_base2_arm18_phase_v1',basket_training_stage='mobile_joint',
            basket_reward_version=self.basket_reward_version,basket_observation_version='mobile51_v1',
            basket_arm_control=self.basket_arm_control,basket_mobile_options=self.mobile_options,
            basket_fixed_protocol='aim_carry_fingers_v1',basket_finger_options=self.finger_options,
            basket_carry_options=self.carry_options,basket_transfer=self.basket_transfer_metadata,
            basket_throw_lr_scale=self.throw_lr_scale,input_normalization_frozen=False,
            training_metrics=dict(agent_steps=int(self.agent_steps),epoch=int(self.epoch_num),
                episode_reward=float(self.episode_rewards.get_mean()),
                episode_success=float(self.episode_success.get_mean()),
                grouped_success={k:float(v.get_mean()) for k,v in self.mobile_group_metrics.items()},
                grouped_counts={k:int(len(v)) for k,v in self.mobile_group_metrics.items()}))
        torch.save(weights,f'{name}.pth')

    def restore_test(self,fn):
        checkpoint=torch.load(fn,map_location=self.device)
        self._validate_mobile_checkpoint(checkpoint)
        if checkpoint.get('joint_actor_version') != 'phase_conditioned_base2_arm18_v1':
            raise ValueError('Use the new jointly trained BasketMobile41 checkpoint, not the frozen version')
        self.model.load_state_dict(checkpoint['model'],strict=True)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'],strict=True)
        self.value_mean_std.load_state_dict(checkpoint['value_mean_std'],strict=True)
        self.basket_transfer_metadata=checkpoint.get('basket_transfer',{})
