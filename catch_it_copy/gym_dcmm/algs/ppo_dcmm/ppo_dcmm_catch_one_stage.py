import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import math
import time
import torch
import torch.distributed as dist

import wandb

import numpy as np
from omegaconf import OmegaConf

from .experience import ExperienceBuffer
from .models_track import ActorCritic
from .utils import AverageScalarMeter, RunningMeanStd
from .ppo_dcmm_track import terminal_metrics

from tensorboardX import SummaryWriter

class PPO_Catch_OneStage(object):
    def __init__(self, env, output_dif, full_config):
        self.rank = -1
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = int(self.ppo_config['num_actors'])
        self.actions_num = self.env.call("act_c_dim")[0]
        self.actions_low = self.env.call("actions_low")[0]
        self.actions_high = self.env.call("actions_high")[0]
        self.obs_shape = (self.env.call("obs_c_dim")[0],)
        self.full_action_dim = self.env.call("act_c_dim")[0]
        self.fixed_basket = (full_config.get('object_motion') in ('basket', 'throw_basket')
                             and bool(full_config.get('basket_fixed_base_training', False)))
        self.basket_arm_control = str(full_config.get('basket_arm_control', 'cartesian'))
        self.mobile_basket = full_config.get('basket_reward_version') == 'basket_mobile_v1'
        self.fixed_physical = bool(full_config.get('basket_fixed_physical',False))
        self.carry_reward = bool(full_config.get('basket_carry_reward',False))
        from gym_dcmm.utils.basket_carry import make_options
        self.carry_options = make_options(full_config.get('basket_carry_options',{}))
        self.finger_reward = bool(full_config.get('basket_finger_reward', False))
        from gym_dcmm.utils.basket_fingers import make_options as finger_options
        self.finger_options = finger_options(full_config.get('basket_finger_options', {}))
        self.basket_training_stage = str(full_config.get('basket_training_stage', 'joint'))
        self.parking_policy = None
        self.pose_eval_override = bool(full_config.get('basket_pose_eval_override',False))
        self.mask_navigation_updates = bool(full_config.get('basket_mask_navigation_updates', False)) and self.basket_training_stage in ('throw','hold')
        self.basket_reward_version = str(full_config.get('basket_reward_version', 'legacy'))
        self.mobile_seed = int(full_config.seed) if int(full_config.seed) >= 0 else None
        self.mobile_base_scale = float(full_config.get('basket_mobile', {}).get('max_speed', .5))
        options = full_config.get('basket_mobile', {})
        self.mobile_options = OmegaConf.to_container(options, resolve=True) if OmegaConf.is_config(options) else dict(options)
        self.basket_transfer_metadata = {}
        self.input_normalization_frozen = bool(full_config.get('basket_freeze_input_normalization',False))
        self.basket_viewer_speed = 0.
        if (self.fixed_basket or self.mobile_basket) and full_config.test and full_config.viewer:
            self.basket_viewer_speed = float(full_config.get('basket_viewer_speed', 1.))
            if self.basket_viewer_speed < 0 or not np.isfinite(self.basket_viewer_speed):
                raise ValueError('basket_viewer_speed must be finite and >= 0')
            if self.basket_viewer_speed and int(full_config.num_envs) != 1:
                raise ValueError('Basket viewer pacing requires num_envs=1')

        if self.fixed_basket:
            # Learn only arm + hand; insert the two zero base commands at execution.
            self.actions_num = self.full_action_dim - 2
        if self.mobile_basket and self.basket_training_stage != 'joint':
            self.actions_num = 2 if self.basket_training_stage == 'parking' else 18
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'separate_value_mlp': self.network_config.get('separate_value_mlp', True),
        }
        print("net_config: ", net_config)
        self.model = ActorCritic(net_config)
        if self.mask_navigation_updates and not self.model.separate_value_mlp:
            raise ValueError('Masked Basket updates require separate actor/value MLPs')
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        if self.mobile_basket and not full_config.test and full_config.get('basket_warm_start'):
            from gym_dcmm.utils.basket_transfer import from_file
            self.basket_transfer_metadata = from_file(self.model, self.running_mean_std,
                full_config.basket_warm_start, self.device)
            self.input_normalization_frozen = True
            print('[basket-transfer] imported arm18 actor; critic fresh; normalization frozen; source=' + self.basket_transfer_metadata['source_name'], flush=True)
        if self.mobile_basket and self.basket_training_stage == 'throw' and not full_config.test:
            from gym_dcmm.utils.basket_parking_policy import FrozenParkingPolicy
            source = torch.load(full_config.basket_parking_checkpoint, map_location=self.device)
            self.parking_policy = FrozenParkingPolicy(source, self.mobile_options, self.device)
            print('[basket-two-stage] parking actor and its normalization frozen; PPO trains arm18 only', flush=True)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config['learning_rate'])
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr, eps=1e-5)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.action_track_denorm = self.ppo_config['action_track_denorm']
        self.action_catch_denorm = self.ppo_config['action_catch_denorm']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scale_value = self.ppo_config['reward_scale_value']
        self.clip_value_loss = self.ppo_config['clip_value_loss']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(
                self.init_lr,
                self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)
        self.episode_test_rewards = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.mobile_metrics = {key: AverageScalarMeter(200) for key in (
            'parking_success', 'qualified_release', 'basket_hit',
            'safe_basket_success', 'robot_dangerous_collision')} if self.mobile_basket else {}
        self.mobile_group_metrics = {key: AverageScalarMeter(200) for key in (
            'rehearsal_success', 'mobile_success')} if self.mobile_basket and self.mobile_options.get('fixed_throw_probability', 0.) and self.basket_training_stage != 'hold' else {}
        self.obs = None
        self.epoch_num = 0
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_shape[0], self.actions_num, self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.max_test_steps = self.ppo_config['max_test_steps']
        self.best_rewards = -10000
        self.best_mobile_success = -1.
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
            'losses/actor_loss': torch.mean(torch.stack(a_losses)).item(),
            'losses/bounds_loss': torch.mean(torch.stack(b_losses)).item(),
            'losses/critic_loss': torch.mean(torch.stack(c_losses)).item(),
            'losses/entropy': torch.mean(torch.stack(entropies)).item(),
            'info/last_lr': self.last_lr,
            'info/e_clip': self.e_clip,
            'info/kl': torch.mean(torch.stack(kls)).item(),
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v
        for key, meter in self.mobile_metrics.items():
            if len(meter):
                log_dict['basket_mobile/' + key] = meter.get_mean()
        for key, meter in self.mobile_group_metrics.items():
            if len(meter):
                log_dict['basket_mobile/' + key] = meter.get_mean()

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def _update_mobile_metrics(self, infos, dones):
        if not self.mobile_basket:
            return
        from gym_dcmm.utils.basket_mobile import terminal_records
        records = terminal_records(infos, dones)
        for key, meter in self.mobile_metrics.items():
            if records:
                meter.update(torch.tensor([float(record.get(key, False)) for record in records],
                                          dtype=torch.float32, device=self.device))
        for key, meter in self.mobile_group_metrics.items():
            rehearsal = key == 'rehearsal_success'
            selected = [float(record.get('safe_basket_success', False)) for record in records
                        if bool(record.get('basket_rehearsal', False)) == rehearsal]
            if selected:
                meter.update(torch.tensor(selected, dtype=torch.float32, device=self.device))

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train(not self.input_normalization_frozen)
        if self.normalize_value:
            self.value_mean_std.train()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()
        reset_obs, _ = self.env.reset(seed=self.mobile_seed) if (self.mobile_basket or self.fixed_basket) else self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.agent_steps = self.batch_size

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            self.storage.data_dict = None

            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (
                self.batch_size) \
                / (time.time() - _last_t)
            _last_t = time.time()
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()
            current_label = f'{mean_rewards:.2f}' if len(self.episode_rewards) else 'n/a'
            best_label = f'{self.best_rewards:.2f}' if self.best_rewards != -10000 else 'n/a'
            info_string = f'Agent Steps: {int(self.agent_steps // 1e3):04}K | FPS: {all_fps:.1f} | ' \
                            f'Last FPS: {last_fps:.1f} | ' \
                            f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                            f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                            f'Current Reward: {current_label} | Best Reward: {best_label}'
            if self.mobile_group_metrics:
                mobile = self.mobile_group_metrics['mobile_success']
                replay = self.mobile_group_metrics['rehearsal_success']
                info_string += (' | Mobile SR: ' + (f'{mobile.get_mean():.2%}' if len(mobile) else 'n/a')
                                + ' | Replay SR: ' + (f'{replay.get_mean():.2%}' if len(replay) else 'n/a'))
            elif self.mobile_basket and self.basket_training_stage == 'hold':
                info_string += ' | HeldThrow SR: ' + (f'{mean_success:.2%}' if len(self.episode_success) else 'n/a')
            if self.fixed_physical:
                info_string += (' | Basket SR: ' if self.carry_reward else ' | QualifiedThrow SR: ') + (f'{mean_success:.2%}' if len(self.episode_success) else 'n/a')
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

            # print("mean_rewards: ", mean_rewards)
            self.writer.add_scalar(
                'metrics/episode_rewards_per_step', mean_rewards, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_lengths_per_step', mean_lengths, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_success_per_step', mean_success, self.agent_steps)
            wandb.log({
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }, step=self.agent_steps)
            ckpt_prefix = f"{self.env.call('object_motion')[0]}_catch"
            checkpoint_name = f'{ckpt_prefix}_ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'{ckpt_prefix}_last'))

            success_meter = self.mobile_group_metrics.get('mobile_success', self.episode_success)
            # Short replay episodes finish first. Their early reward average is
            # not representative of the mixed task or mobile performance.
            reward_eligible = (len(success_meter) >= 50 if self.mobile_group_metrics
                               else (not self.mobile_basket or bool(len(self.episode_rewards))))
            if mean_rewards > self.best_rewards and reward_eligible:
                print(f'save current best reward: {mean_rewards:.2f}')
                # remove previous best file
                prev_best_ckpt = os.path.join(self.nn_dir, f'{ckpt_prefix}_best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'{ckpt_prefix}_best_reward_{mean_rewards:.2f}'))

            if (self.mobile_basket or self.fixed_physical) and len(success_meter) >= 50 and success_meter.get_mean() > self.best_mobile_success:
                self.best_mobile_success = success_meter.get_mean()
                self.save(os.path.join(self.nn_dir, f'{ckpt_prefix}_best_success'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'tracking_mlp': self.model.actor_mlp.state_dict(),
            'tracking_mu': self.model.mu.state_dict(),
            'tracking_sigma': self.model.sigma.data
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        weights['action_layout'] = 'arm6_hand12' if self.fixed_basket else 'base2_arm6_hand12'
        if self.fixed_physical:
            weights['basket_fixed_protocol'] = 'aim_carry_v1' if self.carry_reward else 'physical_settle_forward_v1'
            if self.finger_reward:
                weights['basket_fixed_protocol'] = 'aim_carry_fingers_v1'
                weights['basket_finger_options'] = self.finger_options
            if self.carry_reward:
                weights['basket_carry_options'] = self.carry_options
            weights['input_normalization_frozen'] = self.input_normalization_frozen
            weights['training_metrics'] = dict(agent_steps=int(self.agent_steps),
                episode_success=float(self.episode_success.get_mean()),
                episode_reward=float(self.episode_rewards.get_mean()))
        if self.fixed_basket or self.mobile_basket:
            weights['basket_arm_control'] = getattr(self, 'basket_arm_control', 'cartesian')
        if self.mobile_basket:
            weights['basket_reward_version'] = self.basket_reward_version
            weights['basket_observation_version'] = 'mobile51_v1'
            weights['basket_mobile_options'] = self.mobile_options
            weights['basket_transfer'] = self.basket_transfer_metadata
            weights['input_normalization_frozen'] = self.input_normalization_frozen
            weights['basket_training_stage'] = self.basket_training_stage
            weights['training_metrics'] = dict(agent_steps=int(self.agent_steps),
                epoch=int(self.epoch_num), episode_reward=float(self.episode_rewards.get_mean()),
                episode_success=float(self.episode_success.get_mean()),
                grouped_success={key: float(meter.get_mean()) for key,meter in self.mobile_group_metrics.items()},
                grouped_counts={key: int(len(meter)) for key,meter in self.mobile_group_metrics.items()})
            if self.basket_training_stage == 'parking':
                weights['action_layout'] = 'base2_parking'
            elif self.basket_training_stage == 'throw':
                weights['action_layout'] = 'frozen_base2_arm6_hand12'
                weights['parking_policy'] = self.parking_policy.export()
            elif self.basket_training_stage == 'hold':
                weights['action_layout'] = 'held_arm6_hand12'
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location = self.device)
        self._validate_mobile_checkpoint(checkpoint)
        if self.fixed_basket and checkpoint.get('basket_arm_control', 'cartesian') != self.basket_arm_control:
            raise ValueError('Basket checkpoint action semantics differ: Cartesian and joint-velocity policies cannot be interchanged')
        self._load_compatible_model_state(checkpoint['model'])
        self._load_compatible_running_stats(checkpoint)

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location = self.device)
        self._validate_mobile_checkpoint(checkpoint)
        if self.fixed_basket and checkpoint.get('basket_arm_control', 'cartesian') != self.basket_arm_control:
            raise ValueError('Basket checkpoint action semantics differ: Cartesian and joint-velocity policies cannot be interchanged')
        self._load_compatible_model_state(checkpoint['model'])
        if self.normalize_input:
            self._load_compatible_running_stats(checkpoint)
        if self.mobile_basket:
            self.basket_transfer_metadata = checkpoint.get('basket_transfer', {})
            self.input_normalization_frozen = bool(checkpoint.get('input_normalization_frozen', False))
            if self.basket_training_stage == 'throw':
                from gym_dcmm.utils.basket_parking_policy import FrozenParkingPolicy
                self.parking_policy = FrozenParkingPolicy(checkpoint['parking_policy'], self.mobile_options, self.device)

    def _validate_mobile_checkpoint(self, checkpoint):
        if checkpoint.get('basket_fixed_protocol') == 'aim_carry_fingers_v1':
            if not getattr(self, 'finger_reward', False):
                raise ValueError('Finger-coordinated checkpoint requires basket_finger_sync')
            if checkpoint.get('basket_finger_options') != self.finger_options:
                raise ValueError('Basket finger options differ from the checkpoint')
        if checkpoint.get('basket_fixed_protocol') in ('aim_carry_v1', 'aim_carry_fingers_v1'):
            if not getattr(self,'carry_reward',False):
                raise ValueError('Carry-shaped checkpoint requires basket_aim_carry')
            if checkpoint.get('basket_carry_options') != self.carry_options:
                raise ValueError('Carry reward options differ from the checkpoint')
        if checkpoint.get('basket_fixed_protocol') and not getattr(self,'fixed_physical',False):
            raise ValueError('This checkpoint requires basket_physical_arm: physical preparation and qualified release')
        stage = getattr(self, 'basket_training_stage', 'joint')
        layout = {'joint': 'base2_arm6_hand12', 'parking': 'base2_parking',
                  'throw': 'frozen_base2_arm6_hand12', 'hold':'held_arm6_hand12'}[stage]
        if self.mobile_basket and (checkpoint.get('basket_reward_version') != 'basket_mobile_v1'
                or checkpoint.get('basket_arm_control') != 'joint_velocity'
                or checkpoint.get('basket_observation_version') != 'mobile51_v1'
                or checkpoint.get('action_layout') != layout
                or checkpoint.get('basket_training_stage', 'joint') != stage):
            raise ValueError('Mobile Basket requires its own 51-observation/20-action checkpoint; fixed-base weights need an explicit migration')
        if self.mobile_basket and checkpoint.get('basket_mobile_options') is not None:
            from gym_dcmm.utils.basket_mobile import DEFAULTS
            saved_options = dict(DEFAULTS, **checkpoint['basket_mobile_options'])
            current_options = dict(DEFAULTS, **self.mobile_options)
            # Evaluation may select pure mobile or pure fixed-task rehearsal.
            saved_options.pop('fixed_throw_probability', None)
            current_options.pop('fixed_throw_probability', None)
            if stage == 'hold':
                # Delayed-throw evaluation must be able to test wait-time bins.
                saved_options.pop('hold_time_range', None)
                current_options.pop('hold_time_range', None)
                if getattr(self,'pose_eval_override',False):
                    changed={}
                    for key in ('hold_arm_offset_deg','hold_hand_flex','hold_thumb_flex','hold_ball_local'):
                        if saved_options.get(key)!=current_options.get(key):
                            changed[key]=dict(training=saved_options.get(key),evaluation=current_options.get(key))
                        saved_options.pop(key,None)
                        current_options.pop(key,None)
                    print('[basket-pose-evaluation] explicit pose changes: '+str(changed),flush=True)
            if saved_options != current_options:
                raise ValueError('Mobile checkpoint task options differ; use its training preset/options for evaluation')

    def _load_compatible_running_stats(self, checkpoint):
        """兼容 Tracking checkpoint (18维) 和 Catching checkpoint (30维) 的 running_mean_std"""
        if 'running_mean_std' not in checkpoint:
            print("[checkpoint] running_mean_std not found, using fresh stats")
            return
        ckpt_shape = checkpoint['running_mean_std']['running_mean'].shape
        curr_shape = self.running_mean_std.running_mean.shape
        if ckpt_shape == curr_shape:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        else:
            print(f"[checkpoint] running_mean_std shape mismatch: checkpoint {ckpt_shape} vs model {curr_shape}, using fresh stats")

    def _load_compatible_model_state(self, checkpoint_state):
        if self.fixed_basket or getattr(self, 'mobile_basket', False):
            # Partial row copying would reinterpret old base actions as arm actions.
            self.model.load_state_dict(checkpoint_state, strict=True)
            return
        model_state = self.model.state_dict()
        loaded, expanded, skipped = [], [], []
        for name, value in checkpoint_state.items():
            if name not in model_state:
                skipped.append(name)
                continue

            target = model_state[name]
            if value.shape == target.shape:
                model_state[name] = value
                loaded.append(name)
            elif value.ndim == target.ndim and value.ndim > 0 and value.shape[1:] == target.shape[1:]:
                merged = target.clone()
                n = min(value.shape[0], target.shape[0])
                merged[:n] = value[:n]
                model_state[name] = merged
                expanded.append(f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")
            else:
                skipped.append(f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")

        self.model.load_state_dict(model_state)
        if expanded:
            print("[checkpoint] Partially loaded resized tensors:")
            for item in expanded:
                print("  ", item)
        if skipped:
            print("[checkpoint] Skipped incompatible tensors:")
            for item in skipped:
                print("  ", item)

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs = self.storage[i]

                valid_action = obs[:,47] > .5 if self.mask_navigation_updates else None
                has_valid_action = valid_action is None or bool(valid_action.any())

                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                if self.clip_value_loss:
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(-self.e_clip, self.e_clip)
                    value_losses = (values - returns) ** 2
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    c_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    c_loss = (values - returns) ** 2
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                if valid_action is not None:
                    from gym_dcmm.utils.basket_policy_mask import masked_mean
                    a_loss, entropy, b_loss = [masked_mean(loss, valid_action) for loss in (a_loss, entropy, b_loss)]
                    c_loss = c_loss.mean()
                else:
                    a_loss, c_loss, entropy, b_loss = [
                        torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef \
                    + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                if not has_valid_action:
                    from gym_dcmm.utils.basket_policy_mask import clear_actor_gradients
                    clear_actor_gradients(self.model)

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    if valid_action is None:
                        kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)
                    elif has_valid_action:
                        kl_dist = policy_kl(mu.detach()[valid_action], sigma.detach()[valid_action],
                                            old_mu[valid_action], old_sigma[valid_action])
                    else:
                        kl_dist = mu.new_zeros(())

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                if has_valid_action:
                    ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls)) if ep_kls else self.model.sigma.new_zeros(())
            kls.append(av_kls)

            if self.lr_schedule == 'kl' and ep_kls:
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = self.adjust_learning_rate_cos(mini_ep)

            for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.last_lr

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls
    
    def obs2tensor(self, obs):
        # Map the step result to tensor
        _parts = [
            obs["base"]["v_lin_2d"],
            obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
            obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
        ]
        if "hand" in obs:
            _parts.append(obs["hand"])
        if "basket" in obs:
            _parts.append(obs["basket"]["rel_pos3d"])
            _parts.append(obs["basket"]["target_rel_pos2d"])
        if "mobile" in obs:
            _parts.append(obs['mobile'])
        obs_array = np.concatenate(tuple(_parts), axis=1)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        return obs_tensor

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        stage = getattr(self, 'basket_training_stage', 'joint')
        if getattr(self, 'mobile_basket', False) and stage == 'parking':
            return dict(base=actions[:, :2]*self.mobile_base_scale,
                        arm=np.zeros((len(actions), 6)), hand=np.zeros((len(actions), 12)))
        if getattr(self, 'mobile_basket', False) and stage == 'throw':
            base = self.parking_policy.actions(self.obs['obs']).cpu().numpy()*self.mobile_base_scale
            return dict(base=base, arm=actions[:, :6]*self.action_catch_denorm[1],
                        hand=actions[:, 6:18]*self.action_catch_denorm[2])
        if getattr(self, 'mobile_basket', False) and stage == 'hold':
            return dict(base=np.zeros((len(actions),2)),
                        arm=actions[:,:6]*self.action_catch_denorm[1],
                        hand=actions[:,6:18]*self.action_catch_denorm[2])
        if self.fixed_basket:
            # Rollout code pads to the environment width on the right; ignore padding.
            return dict(base=np.zeros((len(actions), 2), dtype=actions.dtype),
                        arm=actions[:, :6] * self.action_catch_denorm[1],
                        hand=actions[:, 6:18] * self.action_catch_denorm[2])
        # De-normalize the actions
        if self.env.call('task')[0] == 'Tracking':
            base_tensor = actions[:, :2] * self.action_track_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_track_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_track_denorm[2]
        else:
            base_tensor = actions[:, :2] * self.action_catch_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_catch_denorm[2]
        if getattr(self, 'mobile_basket', False):
            base_tensor = actions[:, :2] * self.mobile_base_scale
        actions_dict = {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor
        }
        return actions_dict
    
    def model_act(self, obs_dict, inference=False):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
        }
        if not inference:
            res_dict = self.model.act(input_dict)
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        else:
            res_dict = {}
            res_dict['actions'] = self.model.act_inference(input_dict)
        return res_dict

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # Collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # Do env step
            # Clamp the actions of the action space
            actions = res_dict['actions']
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scale_value * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            # print("self.dones: ", self.dones)
            done_indices = self.dones.nonzero(as_tuple=False)
            # print("done_indices: ", done_indices)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            successes, _ = terminal_metrics(infos, dones, truncates)
            self.episode_success.update(torch.tensor(successes, dtype=torch.float32, device=self.device)[done_indices])
            self._update_mobile_metrics(infos, dones)
            assert isinstance(infos, dict), 'Info Should be a Dict'
            # print("infos: ", infos)
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps = (self.agent_steps + self.batch_size)
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()
        if self.mask_navigation_updates:
            data = self.storage.data_dict
            valid = data['obses'][:,47] > .5
            self.extra_info['basket_mobile/active_action_fraction'] = valid.float().mean().item()
            if valid.any():
                selected = data['advantages'][valid]
                data['advantages'] = (data['advantages']-selected.mean())/selected.std(unbiased=False).clamp_min(1e-8)
            else:
                data['advantages'].zero_()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def play_test_steps(self):
        for _ in range(self.horizon_length):
            if self.basket_viewer_speed:
                clock_env = self.env.unwrapped.envs[0].unwrapped
                wall_start = time.perf_counter()
                sim_start = float(clock_env.Dcmm.data.time)

            res_dict = self.model_act(self.obs, inference=True)
            # Do env step
            # Clamp the actions of the action space 
            actions = res_dict['actions']
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            if self.basket_viewer_speed:
                # Reset-only steps may move simulation time backwards: do not sleep.
                elapsed_sim = max(0., float(clock_env.Dcmm.data.time) - sim_start)
                delay = elapsed_sim / self.basket_viewer_speed - (time.perf_counter() - wall_start)
                if delay > 0:
                    time.sleep(delay)

            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_test_rewards.update(self.current_rewards[done_indices])
            self.episode_test_lengths.update(self.current_lengths[done_indices])
            successes, _ = terminal_metrics(infos, dones, truncates)
            self.episode_test_success.update(torch.tensor(successes, dtype=torch.float32, device=self.device)[done_indices])
            self._update_mobile_metrics(infos, dones)
            if self.mobile_basket or self.fixed_basket:
                self.mobile_test_episodes += int(np.count_nonzero(dones))
            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            if (self.mobile_basket or self.fixed_basket) and self.mobile_test_episodes >= self.ppo_config['test_num_episodes']:
                break
        
        res_dict = self.model_act(self.obs)
        self.agent_steps = (self.agent_steps + self.batch_size)

    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset(seed=self.mobile_seed) if self.mobile_basket else self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.test_steps = self.batch_size
        self.mobile_test_episodes = 0

        while self.test_steps < self.max_test_steps:
            self.play_test_steps()
            self.storage.data_dict = None
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            mean_success = self.episode_test_success.get_mean()
            print("## Sample Length %d ##" % len(self.episode_test_rewards))
            print("mean_rewards: ", mean_rewards)
            print("mean_lengths: ", mean_lengths)
            print("mean_success: ", mean_success)
            if self.mobile_basket:
                print('basket_mobile:', {key: meter.get_mean() for key, meter in dict(self.mobile_metrics, **self.mobile_group_metrics).items()})
            self.test_steps += self.horizon_length
            if (self.mobile_basket or self.fixed_basket) and self.mobile_test_episodes >= self.ppo_config['test_num_episodes']:
                break
            # wandb.log({
            #     'metrics/episode_test_rewards': mean_rewards,
            #     'metrics/episode_test_lengths': mean_lengths,
            # }, step=self.agent_steps)
    

    def adjust_learning_rate_cos(self, epoch):
        lr = self.init_lr * 0.5 * (
            1. + math.cos(
                math.pi * (self.agent_steps + epoch / self.mini_epochs_num) / self.max_agent_steps))
        return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class LinearScheduler:
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)
