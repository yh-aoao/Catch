import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import time
import torch
import numpy as np
import wandb
from tensorboardX import SummaryWriter

from .replay_buffer import ReplayBuffer
from .models_track import SACActor, QNetwork
from .utils import AverageScalarMeter, RunningMeanStd


class SAC_Track(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.sac_config = full_config.train.sac

        # ---- build environment ----
        self.env = env
        self.num_actors = int(self.sac_config['num_actors'])
        print("num_actors: ", self.num_actors)
        self.actions_num = self.env.call("act_t_dim")[0]
        print("actions_num: ", self.actions_num)
        self.obs_shape = (self.env.call("obs_t_dim")[0],)
        self.full_action_dim = self.env.call("act_c_dim")[0]

        # ---- SAC 超参数 ----
        self.gamma = self.sac_config['gamma']
        self.tau = self.sac_config['tau']
        self.reward_scale_value = self.sac_config['reward_scale_value']
        self.batch_size = self.sac_config['batch_size']
        self.start_steps = self.sac_config['start_steps']
        self.updates_per_step = self.sac_config['updates_per_step']
        self.auto_entropy_tuning = self.sac_config['auto_entropy_tuning']
        self.target_entropy_scale = self.sac_config['target_entropy_scale']
        self.truncate_grads = self.sac_config['truncate_grads']
        self.grad_norm = self.sac_config['grad_norm']
        self.normalize_input = self.sac_config['normalize_input']
        self.action_track_denorm = self.sac_config['action_track_denorm']
        self.action_catch_denorm = self.sac_config['action_catch_denorm']
        self.horizon_length = self.sac_config['horizon_length']

        # ---- Models ----
        net_units = self.network_config.mlp.units
        log_std_min = self.sac_config['log_std_min']
        log_std_max = self.sac_config['log_std_max']

        self.actor = SACActor(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        ).to(self.device)

        self.q1 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2 = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self.q1_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)
        self.q2_target = QNetwork(
            obs_dim=self.obs_shape[0],
            action_dim=self.actions_num,
            units=net_units,
        ).to(self.device)

        self._hard_update(self.q1_target, self.q1)
        self._hard_update(self.q2_target, self.q2)

        # ---- 温度参数 ----
        target_entropy = -self.actions_num * self.target_entropy_scale
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        # ---- Observation Normalization ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)

        # ---- Output Dir ----
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Optimizers ----
        self.init_lr = float(self.sac_config['learning_rate'])
        self.alpha_lr = float(self.sac_config['alpha_lr'])

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), self.init_lr, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            self.init_lr, eps=1e-5)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], self.alpha_lr, eps=1e-5)

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(
            capacity=self.sac_config['replay_buffer_size'],
            obs_dim=self.obs_shape[0],
            act_dim=self.actions_num,
            device=self.device,
        )

        # ---- Logging & Metrics ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer
        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)
        self.episode_test_rewards = AverageScalarMeter(self.sac_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(self.sac_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(self.sac_config['test_num_episodes'])

        self.obs = None
        self.epoch_num = 0

        batch_size = self.num_actors
        self.current_rewards = torch.zeros(
            (batch_size, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones(
            (batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.sac_config['max_agent_steps']
        self.max_test_steps = self.sac_config['max_test_steps']
        self.best_rewards = -10000

        self.data_collect_time = 0
        self.rl_train_time = 0

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    @staticmethod
    def _soft_update(target, source, tau):
        for target_param, source_param in zip(
            target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1.0 - tau) * target_param.data + tau * source_param.data)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = \
            self.replay_buffer.sample(self.batch_size)

        rewards_batch = rewards_batch * self.reward_scale_value

        # ---- Q 网络更新 ----
        with torch.no_grad():
            next_action, next_log_prob, _, _ = self.actor.act(next_obs_batch)
            target_q1 = self.q1_target(next_obs_batch, next_action)
            target_q2 = self.q2_target(next_obs_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = target_q - self.alpha * next_log_prob
            q_target = rewards_batch + (1.0 - dones_batch) * self.gamma * target_value

        q1_pred = self.q1(obs_batch, actions_batch)
        q2_pred = self.q2(obs_batch, actions_batch)
        q1_loss = torch.nn.functional.mse_loss(q1_pred, q_target.detach())
        q2_loss = torch.nn.functional.mse_loss(q2_pred, q_target.detach())
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.truncate_grads:
            q_params = list(self.q1.parameters()) + list(self.q2.parameters())
            torch.nn.utils.clip_grad_norm_(q_params, self.grad_norm)
        self.q_optimizer.step()

        # ---- Policy 更新 ----
        new_action, new_log_prob, _, _ = self.actor.evaluate(obs_batch)
        q1_new = self.q1(obs_batch, new_action)
        q2_new = self.q2(obs_batch, new_action)
        q_min = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * new_log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.truncate_grads:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()

        # ---- Alpha 更新 ----
        if self.auto_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (new_log_prob.detach() + self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        # ---- 软更新 Target Q ----
        self._soft_update(self.q1_target, self.q1, self.tau)
        self._soft_update(self.q2_target, self.q2, self.tau)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha.item(),
        }

    def obs2tensor(self, obs):
        task = self.env.call('task')[0]
        if task == 'Catching':
            obs_array = np.concatenate((
                obs["base"]["v_lin_2d"],
                obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"],
                obs["arm"]["ee_v_lin_3d"],
                obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
                obs["hand"],
            ), axis=1)
        else:
            obs_array = np.concatenate((
                obs["base"]["v_lin_2d"],
                obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"],
                obs["arm"]["ee_v_lin_3d"],
                obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
            ), axis=1)
        return torch.tensor(obs_array, dtype=torch.float32).to(self.device)

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        task = self.env.call('task')[0]
        if task == 'Tracking':
            base_tensor = actions[:, :2] * self.action_track_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_track_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_track_denorm[2]
        else:
            base_tensor = actions[:, :2] * self.action_catch_denorm[0]
            arm_tensor = actions[:, 2:8] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 8:] * self.action_catch_denorm[2]
        return {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor,
        }

    def model_act(self, obs_dict, inference=False):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        if inference:
            actions = self.actor.act_inference(processed_obs)
            return {'actions': actions}
        else:
            action, log_prob, mu, sigma = self.actor.act(processed_obs)
            return {
                'actions': action,
                'log_probs': log_prob,
                'mus': mu,
                'sigmas': sigma,
            }

    def play_step(self, random_action=False):
        obs_tensor = self.obs['obs']
        next_obs_list = []
        action_list = []

        for i in range(self.num_actors):
            if random_action:
                action = torch.rand(self.actions_num, device=self.device) * 2.0 - 1.0
            else:
                single_obs = obs_tensor[i:i+1]
                processed_obs = self.running_mean_std(single_obs)
                action, _, _, _ = self.actor.act(processed_obs)
                action = action.squeeze(0)

            action = torch.clamp(action, -1.0, 1.0)
            action_list.append(action)

        actions_tensor = torch.stack(action_list)
        actions_padded = torch.nn.functional.pad(
            actions_tensor, (0, self.full_action_dim - self.actions_num), value=0)
        actions_dict = self.action2dict(actions_padded)

        obs, r, terminates, truncates, infos = self.env.step(actions_dict)
        next_obs_tensor = self.obs2tensor(obs)

        r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
        dones = terminates | truncates
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        for i in range(self.num_actors):
            self.replay_buffer.add(
                obs_tensor[i],
                actions_tensor[i],
                r_tensor[i].unsqueeze(0),
                next_obs_tensor[i],
                dones_tensor[i].unsqueeze(0),
            )

        self.obs = {'obs': next_obs_tensor}

        self.current_rewards += r_tensor.unsqueeze(1)
        self.current_lengths += 1
        done_indices = dones_tensor.nonzero(as_tuple=False)
        self.episode_rewards.update(self.current_rewards[done_indices])
        self.episode_lengths.update(self.current_lengths[done_indices])
        success_values = torch.tensor(
            truncates, dtype=torch.float32, device=self.device)[done_indices]
        self.episode_success.update(success_values)

        for k, v in infos.items():
            if isinstance(v, float) or isinstance(v, int) or \
               (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                self.extra_info[k] = v

        not_dones = 1.0 - dones_tensor.unsqueeze(1)
        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * (1.0 - dones_tensor)

    def set_eval(self):
        self.actor.eval()
        if self.normalize_input:
            self.running_mean_std.eval()

    def set_train(self):
        self.actor.train()
        if self.normalize_input:
            self.running_mean_std.train()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()

        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.agent_steps = self.num_actors

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            self.set_eval()

            _collect_t = time.time()
            for _ in range(self.horizon_length):
                random_action = self.agent_steps < self.start_steps
                self.play_step(random_action=random_action)
                self.agent_steps += self.num_actors

            self.data_collect_time += (time.time() - _collect_t)

            _train_t = time.time()
            self.set_train()
            loss_info = None
            for _ in range(self.horizon_length * self.updates_per_step):
                loss_info = self.update()
            self.rl_train_time += (time.time() - _train_t)

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (self.horizon_length * self.num_actors) / (time.time() - _last_t)
            _last_t = time.time()

            info_string = (
                f'Agent Steps: {int(self.agent_steps // 1e3):04}K | '
                f'FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | '
                f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                f'Current Best: {self.best_rewards:.2f}'
            )
            print(info_string)

            self._write_stats(loss_info)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()

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

            ckpt_prefix = f"{self.env.call('object_motion')[0]}_track_sac"
            checkpoint_name = (
                f'{ckpt_prefix}_ep_{self.epoch_num}_'
                f'step_{int(self.agent_steps // 1e6):04}m_'
                f'reward_{mean_rewards:.2f}'
            )

            save_freq = self.sac_config['save_frequency']
            if save_freq > 0:
                if (self.epoch_num % save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'{ckpt_prefix}_last'))

            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                prev_best_ckpt = os.path.join(
                    self.nn_dir, f'{ckpt_prefix}_best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(
                    self.nn_dir, f'{ckpt_prefix}_best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def _write_stats(self, loss_info):
        if loss_info is None:
            return
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,
            'losses/q1_loss': loss_info['q1_loss'],
            'losses/q2_loss': loss_info['q2_loss'],
            'losses/policy_loss': loss_info['policy_loss'],
            'losses/alpha_loss': loss_info['alpha_loss'],
            'info/alpha': loss_info['alpha'],
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v
        wandb.log(log_dict, step=self.agent_steps)
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def save(self, name):
        weights = {
            'model': self.actor.state_dict(),
            'actor_mlp_t': self.actor.actor_mlp.state_dict(),
            'mu_t': self.actor.mu.state_dict(),
            'log_std_t': self.actor.log_std.state_dict(),
            'running_mean_std': self.running_mean_std.state_dict(),
        }
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self._load_compatible_model_state(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self._load_compatible_model_state(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def _load_compatible_model_state(self, checkpoint_state):
        model_state = self.actor.state_dict()
        loaded, expanded, skipped = [], [], []
        for name, value in checkpoint_state.items():
            if name not in model_state:
                skipped.append(name)
                continue
            target = model_state[name]
            if value.shape == target.shape:
                model_state[name] = value
                loaded.append(name)
            elif value.ndim == target.ndim and value.ndim > 0 and \
                 value.shape[1:] == target.shape[1:]:
                merged = target.clone()
                n = min(value.shape[0], target.shape[0])
                merged[:n] = value[:n]
                model_state[name] = merged
                expanded.append(
                    f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")
            else:
                skipped.append(
                    f"{name}: {tuple(value.shape)} -> {tuple(target.shape)}")
        self.actor.load_state_dict(model_state)
        if expanded:
            print("[checkpoint] Partially loaded resized tensors:")
            for item in expanded:
                print("  ", item)
        if skipped:
            print("[checkpoint] Skipped incompatible tensors:")
            for item in skipped:
                print("  ", item)

    def test(self):
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        test_steps = self.num_actors

        while test_steps < self.max_test_steps:
            self._play_test_steps()
            test_steps += self.horizon_length * self.num_actors
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            mean_success = self.episode_test_success.get_mean()
            print(f"## Sample Length {len(self.episode_test_rewards)} ##")
            print(f"mean_rewards: {mean_rewards}")
            print(f"mean_lengths: {mean_lengths}")
            print(f"mean_success: {mean_success}")

    def _play_test_steps(self):
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            actions = res_dict['actions']
            actions = torch.clamp(actions, -1.0, 1.0)
            actions = torch.nn.functional.pad(
                actions, (0, self.full_action_dim - self.actions_num), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            self.obs = {'obs': self.obs2tensor(obs)}

            r_tensor = torch.tensor(r, dtype=torch.float32, device=self.device)
            self.current_rewards += r_tensor.unsqueeze(1)
            self.current_lengths += 1
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_test_rewards.update(self.current_rewards[done_indices])
            self.episode_test_lengths.update(self.current_lengths[done_indices])
            success_values = torch.tensor(
                truncates, dtype=torch.float32, device=self.device)[done_indices]
            self.episode_test_success.update(success_values)

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
