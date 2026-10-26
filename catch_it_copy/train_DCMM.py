from __future__ import annotations

import hydra
import torch
import os
import random
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from gym_dcmm.utils.util import omegaconf_to_dict
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track
import gymnasium as gym
import gym_dcmm
import datetime
import pytz
# os.environ['MUJOCO_GL'] = 'egl'
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    if config.test_object != 'default':
        if not config.test or config.object_motion not in ('throw', 'roll', 'bounce') or config.object_eval:
            raise ValueError('test_object requires test=True, object_eval=false, throw/roll/bounce')
    if config.basket_reward_version in ('launch_v4', 'launch_v5', 'launch_simple_v1'):
        if config.object_motion not in ('basket', 'throw_basket') or config.basket_arm_control != 'joint_velocity' or not config.basket_fixed_base_training or config.task != 'Catching_OneStage':
            raise ValueError('launch_v4/v5/simple requires fixed-base Basket joint_velocity Catching_OneStage')
        if config.test and config.basket_curriculum_stage != 3:
            raise ValueError('Formal Basket testing requires curriculum stage 3')
    if config.object_motion in ('basket', 'throw_basket') and config.basket_arm_control == 'joint_velocity':
        if not config.basket_fixed_base_training or config.task != 'Catching_OneStage':
            raise ValueError('Joint-velocity throw v3 currently requires fixed-base Catching_OneStage')
        if abs(float(config.train.ppo.action_catch_denorm[1]) - .025) > 1e-9:
            raise ValueError('Joint-velocity throw expects arm action de-normalization .025')
    if config.object_motion in ('basket', 'throw_basket') and config.basket_fixed_base_training:
        if config.task != 'Catching_OneStage' or config.checkpoint_tracking:
            raise ValueError('Fixed Basket baseline requires Catching_OneStage and no checkpoint_tracking')
        print('[basket-baseline] fixed base command, fixed target, policy=arm6+hand12; reward=joint_throw', flush=True)
    if config.object_motion in ('bounce', 'tan', '\u5f39'):
        # Catching \u7528 GitHub 5fe75d5f\uff1bTracking \u7528\u672c\u5730 2026-07-22 (58.18)
        from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as TwoStageAgent
        from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage as OneStageAgent
        if config.task == 'Tracking':
            from gym_dcmm.algs.ppo_dcmm_july22.ppo_dcmm_track import PPO_Track as TrackingAgent
            print('[bounce-track-baseline] snapshot=2026-07-22 (58.18) '
                  'environment=DcmmVecEnv_bounce_july22.py PPO=ppo_dcmm_july22.track success=touch_only; catch=5fe75d5f', flush=True)
        else:
            from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_track import PPO_Track as TrackingAgent
            print('[bounce-catch-baseline] commit=5fe75d5f1e151c1fdc686fbf8f104db3870b2844 '
                  'environment=DcmmVecEnv.py::BounceEnv config=original success=environment_flag', flush=True)
    elif config.object_motion == 'roll':
        from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_track import PPO_Track as TrackingAgent
        from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage as TwoStageAgent
        from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage as OneStageAgent
        print('[roll-baseline] task={} environment/config={} PPO=645edc4 (unchanged since 07c176f)'.format(
            config.task, 'DcmmVecEnv.py::RollTrackingEnv (07c176f)' if config.task == 'Tracking' else 'current_catching'), flush=True)
    else:
        TrackingAgent, TwoStageAgent, OneStageAgent = PPO_Track, PPO_Catch_TwoStage, PPO_Catch_OneStage
    if str(config.basket_control_probe) != 'off' and (not config.test or config.object_motion not in ('basket', 'throw_basket')):
        raise ValueError('basket_control_probe is only for Basket test=True, never policy training')
    torch.multiprocessing.set_start_method('spawn')
    config.test = config.test
    model_path = None
    if config.task == 'Tracking' and config.checkpoint_tracking:
        config.checkpoint_tracking = to_absolute_path(config.checkpoint_tracking)
        model_path = config.checkpoint_tracking
    elif (config.task == 'Catching_TwoStage' \
        or config.task == 'Catching_OneStage') \
        and config.checkpoint_catching:
        config.checkpoint_catching = to_absolute_path(config.checkpoint_catching)
        model_path = config.checkpoint_catching

    # use the device for rl
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'
    # config.seed = random.seed(config.seed)
    import numpy as np

    # 设置随机种子
    if config.seed != -1:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if config.device_id >= 0:  # 如果使用 GPU
            torch.cuda.manual_seed_all(config.seed)
        # 启用 PyTorch 确定性模式（可选，会影响性能）
        if config.torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    # Create and wrap the environment
    env_name = 'gym_dcmm/SlideWorld-v0' if config.object_motion == 'slide' else 'gym_dcmm/DcmmVecWorld-v0'
    if config.test_object != 'default':
        env_name = 'gym_dcmm/ObjectTest-v0'
    task = 'Tracking' if config.task == 'Tracking' else 'Catching'
    print("config.num_envs: ", config.num_envs)
    # Single-environment evaluation keeps Viewer and exceptions in the main process.
    vector_mode = 'sync' if config.test and int(config.num_envs) == 1 else 'async'
    requested_vector_mode = str(config.get('vector_mode', 'auto'))
    if requested_vector_mode not in ('auto', 'sync', 'async'):
        raise ValueError('vector_mode must be auto, sync or async')
    if requested_vector_mode != 'auto':
        vector_mode = requested_vector_mode
    print('[environment] vector_mode=' + vector_mode, flush=True)

    env = gym.make_vec(env_name, num_envs=int(config.num_envs), vectorization_mode=vector_mode,
                    task=task, camera_name=["top"],
                    render_per_step=False, render_mode = "rgb_array",
                    object_name = "object",
                    img_size = config.train.ppo.img_dim,
                    imshow_cam = config.imshow_cam, 
                    viewer = config.viewer,
                    print_obs = False, print_info = False,
                    print_reward = False, print_ctrl = False,
                    print_contacts = False, object_eval = config.object_eval,
                    env_time = 5.0 if config.object_motion == 'slide' else 2.5, steps_per_policy = 20,object_motion=config.object_motion,
                    bounce_physics=config.bounce_physics,
                    bounce_launch=config.bounce_launch, bounce_log=config.bounce_log,
                    basket_log=config.basket_log, roll_log=config.roll_log,
                    **({'basket_self_collision_debug': bool(config.basket_self_collision_debug),
                        'basket_fixed_base_training': config.basket_fixed_base_training, 'basket_control_probe': str(config.basket_control_probe),
                        'basket_arm_control': str(config.basket_arm_control),
                        'basket_reward_version': str(config.basket_reward_version),
                        'basket_curriculum_stage': int(config.basket_curriculum_stage)}
                       if config.object_motion in ('basket', 'throw_basket') else {}),
                    **({'test_object': str(config.test_object)} if config.test_object != 'default' else {}),
                    **({'slide_config': to_absolute_path(config.slide_config) if config.slide_config else None,
                        'slide_log': config.slide_log} if config.object_motion == 'slide' else {}))

    output_dif = os.path.join('outputs', config.output_name)
    # Get the local date and time
    local_tz = pytz.timezone('Asia/Shanghai')
    current_datetime = datetime.datetime.now().astimezone(local_tz)
    current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H:%M:%S")
    output_dif = os.path.join(output_dif, current_datetime_str)
    os.makedirs(output_dif, exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dif, 'resolved_config.yaml'), resolve=True)

    PPO = TrackingAgent if config.task == 'Tracking' else \
          TwoStageAgent if config.task == 'Catching_TwoStage' else \
          OneStageAgent
    agent = PPO(env, output_dif, full_config=config)

    cprint('Start Training/Testing the Agent', 'green', attrs=['bold'])
    if config.test:
        if model_path:
            print("checkpoint loaded")
            agent.restore_test(model_path)
        print("testing")
        agent.test()
    else:
        # connect to wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.output_name,
            config=omegaconf_to_dict(config),
            mode=config.wandb_mode
        )

        agent.restore_train(model_path)
        agent.train()

        # close wandb
        wandb.finish()

if __name__ == '__main__':
    main()
