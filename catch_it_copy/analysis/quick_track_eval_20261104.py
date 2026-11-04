"""Use production setup/policy, count exactly 20 completed single-env episodes."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gym_dcmm.algs.ppo_dcmm_645edc4 import ppo_dcmm_track as roll
from gym_dcmm.algs.ppo_dcmm_july22 import ppo_dcmm_track as bounce

@torch.no_grad()
def test(self):
    self.set_eval()
    obs, _ = self.env.reset()
    records = []
    length = 0
    reward = 0.
    for _ in range(20000):
        actions = self.model_act({'obs': self.obs2tensor(obs)}, inference=True)['actions'].clamp(-1, 1)
        actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.shape[1]))
        obs, r, term, trunc, infos = self.env.step(self.action2dict(actions))
        length += 1
        reward += float(r[0])
        done = term | trunc
        if not done[0]:
            continue
        metrics = roll.terminal_metrics if os.environ['EVAL_MOTION'] == 'roll' else bounce.terminal_metrics
        success, reasons = metrics(infos, done, trunc)
        final = infos.get('final_info', [None])[0] or {}
        records.append(dict(episode=len(records), success=bool(success[0]),
                            reason=str(reasons[0]), steps=length, reward=reward,
                            success_version=final.get('success_version'),
                            legacy_success=bool(final.get('success', False))))
        print('QUICK_EPISODE', json.dumps(records[-1]), flush=True)
        length = 0
        reward = 0.
        if len(records) == 20:
            break
    if len(records) != 20:
        raise RuntimeError('Incomplete evaluation')
    result = dict(motion=os.environ['EVAL_MOTION'], object=os.environ['EVAL_OBJECT'],
                  seed=123, episodes=len(records), successes=sum(x['success'] for x in records),
                  records=records, checkpoint=os.environ['EVAL_CHECKPOINT'])
    Path(os.environ['EVAL_RESULT']).write_text(json.dumps(result, indent=2))
    print('QUICK_RESULT', result['successes'], '/', len(records), flush=True)

roll.PPO_Track.test = test
bounce.PPO_Track.test = test
from train_DCMM import main
main()
