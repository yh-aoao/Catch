"""Evaluate current ball tests and legacy Bounce controls without changing training code.

Run in catch_it_copy with PYTHONPATH=. using the dcmm Python environment.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import time

import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch

from ycb_objects.evaluate_trained_balls import PolicyEnvironment, batched, wilson
from ycb_objects.generalization_probe import ContactRecorder, physical_parameters

ROOT = Path(__file__).resolve().parents[1] / 'catch_it_copy'


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reported_success(task, info, truncated):
    """Use the same terminal metric selector as the actual Roll PPO viewer."""
    if task != 'roll':
        return bool(info['success'])
    from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_track import terminal_metrics
    assert 'roll_eval_success' in info, 'Missing current Roll evaluation metric'
    values, _ = terminal_metrics(
        {'success': np.array([info['success']]), 'final_info': np.array([info], dtype=object),
         '_final_info': np.array([True])}, np.array([True]), np.array([truncated]))
    assert bool(values[0]) == bool(info['roll_eval_success'])
    return bool(values[0])


def summary(rows):
    n = len(rows)
    k = sum(r['success'] for r in rows)
    return dict(episodes=n, successes=k, success_rate=k/n,
        wilson_95_interval=wilson(k, n),
        legacy_successes=sum(r.get('legacy_success', r['success']) for r in rows),
        stable_hold_episodes=sum(bool(r.get('roll_stable_hold', False)) for r in rows),
        preparation_failures=sum(r['reason']=='preparation_failed' for r in rows),
        hand_contact_episodes=sum(r.get('hand_contact', False) for r in rows),
        contact_0p3s=sum(r.get('max_contiguous_hand_contact_s', 0)>=.3-1e-6 for r in rows),
        success_with_contact_0p3s=sum(r['success'] and r.get('max_contiguous_hand_contact_s', 0)>=.3-1e-6 for r in rows),
        reasons=dict(Counter(r['reason'] for r in rows)))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['throw', 'roll', 'bounce', 'basket'], required=True)
    parser.add_argument('--object', choices=['original', 'tennis_ball', 'baseball'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    assert args.episodes > 0 and args.seed >= 0
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    cfg.test, cfg.num_envs, cfg.train.ppo.num_actors = True, 1, 1
    cfg.viewer = cfg.imshow_cam = False
    cfg.device_id, cfg.rl_device = -1, 'cpu'
    cfg.checkpoint_tracking = cfg.checkpoint_catching = ''
    name = 'default' if args.object == 'original' else 'ycb:' + args.object
    cfg.test_object = name
    kwargs = dict(task='Catching', object_motion=cfg.object_motion, viewer=False,
                  render_mode=None, imshow_cam=False, object_eval=False,
                  env_time=2.5, steps_per_policy=20)
    if args.task == 'roll':
        from gym_dcmm.envs.roll_physical import make_roll_physical
        from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as Agent
        assert cfg.task == 'Catching_TwoStage' and cfg.object_motion == 'roll'
        assert cfg.roll_physics == 'physical_v1' and cfg.roll_reward_version == 'legacy'
        assert not cfg.roll_wait_guard
        cfg.roll_ball_mode, cfg.roll_ball_weights = 'fixed', None
        material = ROOT / 'ycb_objects/calibration_data/balls.roll_table_priors_v1.json'
        kwargs.update(test_object=name, roll_material_file=str(material),
            roll_reward_version='legacy', roll_speed=OmegaConf.to_container(cfg.roll_speed))
        env = make_roll_physical(**kwargs)
        criterion = dict(name='roll_eval_success:region_retention_v5',
            description='Actual hand contact starts retention; ball has left table, stays clear of table/floor and inside world-vertical palm region at episode end; failure termination overrides. No minimum duration or speed requirement for primary retention metric.',
            selector='Original Roll PPO terminal_metrics(final_info)',
            legacy_speed_limit=env.roll_catch_v_thresh,
            legacy_consecutive_samples=env.roll_catch_N_control, policy_interval=.04)
    elif args.task == 'bounce':
        from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
        from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as Agent
        assert cfg.task == 'Catching_TwoStage' and cfg.object_motion == 'bounce'
        assert cfg.bounce_physics in ('legacy', 'physical_v2') and cfg.get('bounce_grasp_version', 'legacy') == 'legacy'
        cfg.bounce_ball_mode, cfg.bounce_ball_weights = 'fixed', None
        if cfg.bounce_physics == 'legacy':
            from gym_dcmm.envs.DcmmVecEnv import BounceEnv
            assert args.object == 'original' and cfg.bounce_launch == 'legacy'
            material = ROOT/'configs/env/DcmmCfg_bounce_5fe75d5f.py'
            kwargs.update(bounce_physics='legacy', bounce_launch='legacy')
            accepted = inspect.signature(BounceEnv.__init__).parameters
            kwargs = {key: value for key, value in kwargs.items() if key in accepted}
            env = BounceEnv(**kwargs)
        else:
            material = ROOT / 'ycb_objects/calibration_data/balls.three_ball_training_v1.json'
            kwargs.update(test_object=name, bounce_material_file=str(material),
                bounce_launch=cfg.bounce_launch, bounce_v2=OmegaConf.to_container(cfg.bounce_v2))
            env = make_bounce_physical_v2(**kwargs)
        criterion = dict(name='bounce_catch_proximity_v2',
            description='2.5s horizon, no failure, grasp stage, ee distance <0.25m; not strict stable hold')
    elif args.task == 'throw':
        from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
        from gym_dcmm.envs.object_test import make_object_test
        from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as Agent
        assert cfg.task == 'Catching_TwoStage' and cfg.object_motion == 'throw'
        material = ROOT/'configs/env/DcmmCfg.py'
        env = (DcmmVecEnv(**kwargs) if name == 'default' else
               make_object_test(test_object=name, ycb_collision='sphere', **kwargs))
        criterion = dict(name='throw_official',
            description='Original official Throw metric: truncated at 2.5s horizon, including simultaneous termination. Not a strict hand retention metric.',
            physics_note='Existing Throw/object_test route; YCB analytical collision with source mesh inertia and original free-joint settings; not Roll/Bounce physical_v1/v2 materials')
    else:
        from gym_dcmm.envs.DcmmVecEnv import DcmmVecEnv
        from gym_dcmm.envs.basket_object_test import make_basket_object_test
        from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage as Agent
        assert cfg.task == 'Catching_OneStage' and cfg.object_motion == 'throw_basket'
        assert cfg.basket_fixed_physical and cfg.basket_reward_version == 'basket_aim_v1'
        assert not cfg.get('basket_finger_reward', False)
        if args.object == 'tennis_ball':
            cfg.basket_fixed_hand_flex = [1., .9, .9]
        config_data = OmegaConf.to_container(cfg, resolve=True)
        for key in inspect.signature(DcmmVecEnv.__init__).parameters:
            if key.startswith('basket_') and key in config_data:
                kwargs[key] = config_data[key]
        kwargs.update(camera_name=['top'], img_size=config_data['train']['ppo']['img_dim'],
                      render_per_step=False, basket_log=False, basket_mobile_options=None)
        material = ROOT / 'ycb_objects/calibration_data/balls.roll_table_priors_v1.json'
        env = (DcmmVecEnv(**kwargs) if args.object == 'original' else
               make_basket_object_test(test_object=name, **kwargs))
        criterion = dict(name='basket_score',
            description='Physical free-flight hoop crossing plus safety follow-through',
            initial_hand_flex=config_data['basket_fixed_hand_flex'],
            tennis_initial_hand_override=args.object == 'tennis_ball')
    agent = None
    recorder = None
    first_contact = None
    original_contacts = env._get_contacts
    def contacts():
        nonlocal first_contact
        result = original_contacts()
        if recorder is not None:
            had_hand = recorder.had_hand
            recorder.update()
            if recorder.had_hand and not had_hand:
                first_contact = recorder.state()
        return result
    env._get_contacts = contacts
    try:
        agent = Agent(PolicyEnvironment(env), str(args.output/'agent'), cfg)
        agent.restore_test(str(args.checkpoint))
        raw = torch.load(args.checkpoint, map_location='cpu')
        agent.model.load_state_dict(raw['model'], strict=True)
        rms_names = ['running_mean_std'] if args.task == 'basket' else ['running_mean_std_track', 'running_mean_std_hand']
        for key in rms_names:
            getattr(agent, key).load_state_dict(raw[key], strict=True)
        agent.set_eval()
        meta = dict(task=args.task, object=args.object, checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=sha(args.checkpoint), source_config=str(args.config.resolve()),
            source_config_sha256=sha(args.config), effective_config=OmegaConf.to_container(cfg, resolve=True),
            env_kwargs=kwargs, criterion=criterion, seed_start=args.seed, episodes=args.episodes,
            policy='deterministic original PPO, strict weights/RMS, clamp/pad/action2dict unchanged',
            material_path=str(material), material_sha256=sha(material),
            physics=physical_parameters(env), mujoco_version=mujoco.__version__,
            torch_version=torch.__version__, evaluator_sha256=sha(__file__),
            initialization_failures='counted as failures, never retried or omitted')
        (args.output/'provenance.json').write_text(json.dumps(plain(meta), indent=2))
        rows = []
        started = time.monotonic()
        with (args.output/'episodes.jsonl').open('x') as file:
            for seed in range(args.seed, args.seed+args.episodes):
                recorder = None
                first_contact = None
                np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
                try:
                    # Historical Bounce reset has no Gym seed argument. Its
                    # draws use the NumPy/Python RNGs seeded just above.
                    obs, _ = (env.reset() if args.task == 'bounce' and cfg.bounce_physics == 'legacy'
                              else env.reset(seed=seed))
                except ValueError as error:
                    if args.task != 'basket' or 'Physical Basket preparation' not in str(error):
                        raise
                    row = dict(seed=seed, success=False, reason='preparation_failed', error=str(error))
                else:
                    recorder = ContactRecorder(env)
                    initial = recorder.state()
                    episode_physics = physical_parameters(env)
                    preparation = getattr(env, 'basket_preparation', None)
                    reward_sum = 0.
                    for step in range(200):
                        actions = agent.model_act({'obs':agent.obs2tensor(batched(obs))}, inference=True)['actions'].clamp(-1, 1)
                        actions = torch.nn.functional.pad(actions, (0, agent.full_action_dim-actions.shape[1]))
                        command = agent.action2dict(actions)
                        obs, reward, terminated, truncated, info = env.step({k:v[0] for k,v in command.items()})
                        reward_sum += float(reward)
                        assert np.isfinite(env.Dcmm.data.qpos).all() and np.isfinite(reward_sum)
                        if args.task == 'basket':
                            assert float(env.Dcmm.data.ctrl[-1]) == 0., 'Ball force assistance'
                        if terminated or truncated:
                            break
                    else:
                        raise RuntimeError('Episode exceeded 200 policy steps')
                    row = dict(seed=seed, success=reported_success(args.task, info, truncated),
                        legacy_success=bool(info.get('roll_legacy_success', info['success'])),
                        roll_eval_success=info.get('roll_eval_success'),
                        roll_final_hold=info.get('roll_final_hold'),
                        roll_stable_hold=info.get('roll_stable_hold'),
                        roll_caught_once=info.get('roll_caught_once'),
                        roll_evaluation=info.get('roll_eval'),
                        reason=str(info.get('terminated_reason') or env.terminated_reason or 'timeout'),
                        reward=reward_sum, steps=step+1, elapsed=float(info['env_time']),
                        hand_contact=recorder.had_hand, max_contiguous_hand_contact_s=recorder.max_hand_duration,
                        stage=env.stage, initial=initial, final=recorder.state(), preparation=preparation,
                        initial_physics=episode_physics, physics_timestep=float(env.Dcmm.model.opt.timestep),
                        first_hand_contact=first_contact, floor_impacts=recorder.impacts,
                        roll_hold=info.get('roll_hold'), success_version=info.get('success_version'),
                        basket_hit=info.get('basket_hit'), safe_basket_success=info.get('safe_basket_success'))
                file.write(json.dumps(plain(row))+'\n'); file.flush(); rows.append(row)
                if len(rows)%10 == 0 or len(rows) == args.episodes:
                    result = dict(summary(rows), task=args.task, object=args.object, wall_seconds=time.monotonic()-started)
                    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
                    print(json.dumps(result), flush=True)
        sources = {}
        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and Path(path).is_file() and Path(path).suffix == '.py':
                path = Path(path).resolve()
                if ROOT in path.parents:
                    sources[str(path.relative_to(ROOT))] = sha(path)
        (args.output/'source_hashes.json').write_text(json.dumps(sources, indent=2))
        (args.output/'COMPLETE').write_text('completed\n')
    finally:
        if agent is not None:
            agent.writer.close()
        env.close()


if __name__ == '__main__':
    main()
