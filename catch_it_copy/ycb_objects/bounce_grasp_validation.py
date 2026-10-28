"""Compare old/new hand constraints using the same saved Catch policy.

This is a fault regression probe, not an estimate of a newly trained policy's
success. Run from the project with python -m ycb_objects.bounce_grasp_validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from omegaconf import OmegaConf
import torch

from gym_dcmm.envs.bounce_physical_v2 import make_bounce_physical_v2
from gym_dcmm.envs.bounce_grasp import MOVABLE
from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from ycb_objects.evaluate_trained_balls import PolicyEnvironment, batched


@torch.inference_mode()
def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=root.parent/
        'bounce_three_ball_catching_v1_seed123/2026-10-28/11:36:32')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.run/'nn/bounce_catch_best_reward_288.46.pth'
    material = root/'ycb_objects/calibration_data/balls.three_ball_training_v1.json'
    cfg = OmegaConf.load(args.run/'resolved_config.yaml')
    cfg.test, cfg.num_envs, cfg.train.ppo.num_actors = True, 1, 1
    cfg.rl_device, cfg.device_id = 'cpu', -1
    cfg.checkpoint_tracking = cfg.checkpoint_catching = ''
    torch.set_num_threads(1)
    weights = torch.load(checkpoint, map_location='cpu')
    rows = []
    with tempfile.TemporaryDirectory(prefix='bounce_control_probe_') as directory:
        for ball in ('default', 'ycb:tennis_ball', 'ycb:baseball'):
            for version in ('legacy', 'controlled_hold_v1'):
                env = make_bounce_physical_v2(task='Catching', object_motion='bounce',
                    test_object=ball, bounce_launch=cfg.bounce_launch,
                    bounce_v2=OmegaConf.to_container(cfg.bounce_v2),
                    bounce_material_file=material, bounce_grasp_version=version,
                    viewer=False, render_mode=None, object_eval=False,
                    env_time=2.5, steps_per_policy=20)
                agent = PPO_Catch_TwoStage(PolicyEnvironment(env), directory, cfg)
                try:
                    agent.restore_test(str(checkpoint))
                    agent.model.load_state_dict(weights['model'], strict=True)
                    agent.set_eval()
                    model, data = env.Dcmm.model, env.Dcmm.data
                    joint_ids = [int(np.flatnonzero(model.jnt_qposadr == 21+i)[0]) for i in MOVABLE]
                    ranges = model.jnt_range[joint_ids].copy()
                    maximum = dict(actual=0., target=0.)
                    original_contacts = env._get_contacts

                    def record():
                        contacts = original_contacts()
                        q = data.qpos[21+MOVABLE]
                        target = env.Dcmm.target_hand_qpos[MOVABLE]
                        for name, value in (('actual', q), ('target', target)):
                            violation = np.maximum(ranges[:, 0]-value,
                                                   np.maximum(value-ranges[:, 1], 0.))
                            maximum[name] = max(maximum[name], float(violation.max()))
                        return contacts

                    env._get_contacts = record
                    for seed in (0, 1):
                        obs, _ = env.reset(seed=seed)
                        maximum.update(actual=0., target=0.)
                        for step in range(70):
                            actions = agent.model_act({'obs':agent.obs2tensor(batched(obs))},
                                                      inference=True)['actions'].clamp(-1, 1)
                            command = agent.action2dict(actions)
                            obs, _, terminated, truncated, info = env.step({k:v[0] for k,v in command.items()})
                            assert np.isfinite(data.qpos).all()
                            if terminated or truncated:
                                break
                        else:
                            raise RuntimeError('Probe did not terminate')
                        row = dict(ball=ball, version=version, seed=seed, policy_steps=step+1,
                            max_actual_joint_violation_rad=maximum['actual'],
                            max_target_joint_violation_rad=maximum['target'],
                            reason=info.get('terminated_reason') or env.terminated_reason or 'timeout',
                            environment_success=bool(info['success']))
                        if version != 'legacy':
                            assert maximum['target'] <= 1e-9, row
                        rows.append(row)
                        print('[control-probe]', json.dumps(row), flush=True)
                finally:
                    agent.writer.close()
                    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        note='Same legacy Catch policy, same action scale .15rad and seeds; 2ms control samples. New policy must be retrained. Success definitions differ, so rates are not comparable.',
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        rows=rows), indent=2))


if __name__ == '__main__':
    main()
