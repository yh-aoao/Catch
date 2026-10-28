"""Opt-in Catch PPO which protects the pretrained Bounce tracking branch.

The policy, observations, action adapters and PPO loss remain those of the
historical two-stage agent. Only optimization, normalization and resumable
training bookkeeping differ. Legacy agents do not import this module.
"""
import math
import os
import random
import time

import numpy as np
import torch
import wandb

from .ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage


VERSION = 'controlled_hold_v1'
DEFAULTS = dict(track_lr_scale=0.1, freeze_track_steps=500000,
                freeze_track_normalization=True)


def training_options(options=None):
    supplied = dict(options or {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown bounce_grasp_training options: ' + str(unknown))
    cfg = dict(DEFAULTS, **supplied)
    scale = float(cfg['track_lr_scale'])
    steps = cfg['freeze_track_steps']
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('track_lr_scale must be finite and in [0, 1]')
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('freeze_track_steps must be a nonnegative integer')
    if not isinstance(cfg['freeze_track_normalization'], bool):
        raise ValueError('freeze_track_normalization must be boolean')
    cfg['track_lr_scale'] = scale
    return cfg


class _BranchAdam(torch.optim.Adam):
    """Reapply branch rates at every update, including the legacy KL loop.

    The inherited PPO loop writes the same base rate into each parameter group
    after a mini-epoch. Applying rates immediately before step() ensures that
    this cannot silently erase the tracking multiplier.
    """
    def __init__(self, groups, agent):
        self.agent = agent
        super().__init__(groups, lr=agent.init_lr, eps=1e-5)

    def step(self, closure=None):
        self.agent._apply_optimizer_rates()
        return super().step(closure)


class PPO_Catch_Hold(PPO_Catch_TwoStage):
    def __init__(self, env, output_dif, full_config):
        self.hold_training = training_options(full_config.get('bounce_grasp_training'))
        super().__init__(env, output_dif, full_config)
        self._track_parameters = (list(self.model.actor_mlp_t.parameters()) +
                                  list(self.model.mu_t.parameters()) + [self.model.sigma_t])
        track_ids = {id(p) for p in self._track_parameters}
        remaining = [p for p in self.model.parameters() if id(p) not in track_ids]
        self.optimizer = _BranchAdam([
            dict(params=self._track_parameters, branch='track'),
            dict(params=remaining, branch='catch_and_value'),
        ], self)
        self._apply_track_schedule()
        if self.hold_training['freeze_track_normalization']:
            self.running_mean_std_track.eval()

    def load_tracking_model(self, checkpoint_tracking, checkpoint_catching):
        if checkpoint_catching or not checkpoint_tracking:
            return
        checkpoint = torch.load(checkpoint_tracking, map_location=self.device)
        # Resized/partially copied actors would violate the promise to preserve
        # the known-good Track policy. New-version initialization is strict.
        self.model.actor_mlp_t.load_state_dict(checkpoint['tracking_mlp'], strict=True)
        self.model.mu_t.load_state_dict(checkpoint['tracking_mu'], strict=True)
        if checkpoint['tracking_sigma'].shape != self.model.sigma_t.shape:
            raise ValueError('Track sigma shape does not match the Catch tracking branch')
        with torch.no_grad():
            self.model.sigma_t.copy_(checkpoint['tracking_sigma'])
        self.running_mean_std_track.load_state_dict(checkpoint['running_mean_std'], strict=True)
        print('[bounce-hold] pretrained Track weights and normalization loaded strictly')

    def _apply_track_schedule(self):
        frozen = self.agent_steps < self.hold_training['freeze_track_steps']
        for parameter in self._track_parameters:
            parameter.requires_grad_(not frozen)
            if frozen:
                parameter.grad = None
        self.track_frozen = frozen
        self._apply_optimizer_rates()

    def _apply_optimizer_rates(self):
        frozen = self.agent_steps < self.hold_training['freeze_track_steps']
        for group in self.optimizer.param_groups:
            scale = (0.0 if frozen else self.hold_training['track_lr_scale']) \
                    if group['branch'] == 'track' else 1.0
            group['lr'] = float(self.last_lr) * scale

    def set_train(self):
        super().set_train()
        if self.hold_training['freeze_track_normalization']:
            self.running_mean_std_track.eval()
        self._apply_track_schedule()

    def save(self, name):
        # Keep historical inference keys so existing model inspection remains
        # possible, but include the complete optimizer/progress for new resumes.
        numpy_rng = np.random.get_state()
        checkpoint = dict(
            model=self.model.state_dict(),
            running_mean_std_track=self.running_mean_std_track.state_dict(),
            running_mean_std_hand=self.running_mean_std_hand.state_dict(),
            value_mean_std=self.value_mean_std.state_dict(),
            bounce_grasp_version=VERSION,
            bounce_grasp_training=dict(self.hold_training),
            optimizer=self.optimizer.state_dict(),
            agent_steps=int(self.agent_steps), epoch_num=int(self.epoch_num),
            best_rewards=float(self.best_rewards), last_lr=float(self.last_lr),
            scheduler_state=dict(vars(self.scheduler)) if hasattr(self, 'scheduler') else None,
            torch_rng_state=torch.get_rng_state(),
            # Use primitives rather than a NumPy ndarray, so modern PyTorch's
            # default weights_only loader can read our own checkpoints safely.
            numpy_rng_state=(numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2],
                             numpy_rng[3], numpy_rng[4]),
            python_rng_state=random.getstate(),
        )
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        destination = str(name) + '.pth'
        temporary = destination + '.tmp'
        torch.save(checkpoint, temporary)
        os.replace(temporary, destination)

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        for name in ('running_mean_std_track', 'running_mean_std_hand', 'value_mean_std'):
            if name in checkpoint:
                getattr(self, name).load_state_dict(checkpoint[name], strict=True)
        version = checkpoint.get('bounce_grasp_version')
        if version is not None and version != VERSION:
            raise ValueError('Cannot resume a different Bounce grasp version: ' + str(version))
        if version == VERSION:
            saved_options = training_options(checkpoint.get('bounce_grasp_training'))
            if saved_options != self.hold_training:
                raise ValueError('Resume requires the saved bounce_grasp_training options')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.agent_steps = int(checkpoint['agent_steps'])
            self.epoch_num = int(checkpoint['epoch_num'])
            self.best_rewards = float(checkpoint['best_rewards'])
            self.last_lr = float(checkpoint['last_lr'])
            if checkpoint.get('scheduler_state') is not None and hasattr(self, 'scheduler'):
                vars(self.scheduler).update(checkpoint['scheduler_state'])
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
            numpy_rng = checkpoint['numpy_rng_state']
            np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32),
                                 numpy_rng[2], numpy_rng[3], numpy_rng[4]))
            random.setstate(checkpoint['python_rng_state'])
            if torch.cuda.is_available() and checkpoint.get('cuda_rng_state_all'):
                states = checkpoint['cuda_rng_state_all']
                for index, state in enumerate(states[:torch.cuda.device_count()]):
                    torch.cuda.set_rng_state(state.cpu(), device=index)
            print('[bounce-hold] resumed at agent_steps=' + str(self.agent_steps))
        else:
            # Explicit legacy Catch warm start: weights only, never pretend that
            # the old incomplete checkpoint contains resumable optimizer state.
            print('[bounce-hold] legacy Catch weights loaded; new optimizer/progress')
        self._apply_track_schedule()
        if self.hold_training['freeze_track_normalization']:
            self.running_mean_std_track.eval()

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        for name in ('running_mean_std_track', 'running_mean_std_hand'):
            getattr(self, name).load_state_dict(checkpoint[name], strict=True)
        self.set_eval()

    def train(self):
        """The original loop with persistent progress and branch-aware rates.

        Environment state is reset on resume. Progress, optimizer and RNG resume
        correctly, but this is not a bit-exact continuation of an episode.
        """
        started = time.time()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.agent_steps = max(int(self.agent_steps), self.batch_size)
        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            losses = self.train_epoch()
            self.storage.data_dict = None
            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)
            self._apply_optimizer_rates()
            self.extra_info.update(
                bounce_track_frozen=float(self.track_frozen),
                bounce_track_lr=float(self.optimizer.param_groups[0]['lr']),
            )
            self.write_stats(*losses)
            mean_reward = float(self.episode_rewards.get_mean())
            mean_success = float(self.episode_success.get_mean())
            self.writer.add_scalar('metrics/episode_rewards_per_step', mean_reward, self.agent_steps)
            self.writer.add_scalar('metrics/episode_lengths_per_step', self.episode_lengths.get_mean(), self.agent_steps)
            self.writer.add_scalar('metrics/episode_success_per_step', mean_success, self.agent_steps)
            wandb.log(dict(metrics_episode_reward=mean_reward,
                           metrics_episode_success=mean_success), step=self.agent_steps)
            print('[bounce-hold] steps={} frozen_track={} reward={:.3f} success={:.3f}'.format(
                self.agent_steps, self.track_frozen, mean_reward, mean_success), flush=True)
            prefix = '{}_catch'.format(self.env.call('object_motion')[0])
            improved = len(self.episode_rewards) > 0 and mean_reward > self.best_rewards
            if improved:
                previous_best = os.path.join(self.nn_dir, '{}_best_reward_{:.2f}.pth'.format(
                    prefix, self.best_rewards))
                self.best_rewards = mean_reward
                new_best = os.path.join(self.nn_dir, '{}_best_reward_{:.2f}'.format(prefix, mean_reward))
                self.save(new_best)
                if previous_best != new_best + '.pth' and os.path.exists(previous_best):
                    os.remove(previous_best)
            if self.save_freq > 0 and self.epoch_num % self.save_freq == 0:
                self.save(os.path.join(self.nn_dir, '{}_ep_{}'.format(prefix, self.epoch_num)))
            # Each rollout is a recovery point; atomic replacement keeps the
            # preceding checkpoint valid if a save itself is interrupted.
            self.save(os.path.join(self.nn_dir, prefix + '_last'))
        print('[bounce-hold] training complete in {:.1f}s'.format(time.time() - started))

    def test(self):
        # Historical test() never advanced test_steps. This finite evaluator is
        # scoped to the new version and keeps the original policy/test rollout.
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.test_steps = 0
        while self.test_steps < self.max_test_steps:
            self.play_test_steps()
            self.test_steps += self.batch_size
            self.storage.data_dict = None
            print('[bounce-hold-test] steps={} episodes={} success={:.3f}'.format(
                self.test_steps, len(self.episode_test_rewards),
                self.episode_test_success.get_mean()), flush=True)
