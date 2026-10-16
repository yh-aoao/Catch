import copy
import types
from distillation_visual.tests.test_visual import observation
from distillation_visual_aux import environments
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from distillation_visual.tests.test_visual import FakePool as BasePool, FakeTeacher, config as base_config
from distillation_visual_aux import run
from distillation_visual_aux.core import auxiliary_losses, SCHEMA


def config(stage='catch'):
    c = base_config(stage)
    c['auxiliary'] = dict(position_weight=.1, velocity_weight=.1, scales=[1.,1.,1.,5.,5.,5.])
    return c


class FakePool(BasePool):
    def packet(self):
        packet = super().packet()
        packet['object_target'] = np.ones((self.count, 6), np.float32)
        packet['object_valid'] = np.ones((self.count, 6), np.float32)
        packet['object_valid'][self.steps == 0, 3:] = 0
        return packet


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_auxiliary_gradients_reach_both_cameras_and_gru(self):
        for stage in ('track', 'catch'):
            c=config(stage); s=run.make_student(c)
            inputs=(torch.rand(1,2,2,64,64),torch.zeros(1,2,s.proprio_dim),torch.zeros(1,2,s.action_dim))
            a,_,_,pred=s.forward_aux(*inputs)
            loss=auxiliary_losses(pred,torch.ones_like(pred),torch.ones_like(pred),torch.ones(1,2),c['auxiliary'],torch.tensor([2.]))
            (loss['position']+loss['velocity']).backward()
            for param in [s.encoders[0][0][0].weight,s.encoders[1][0][0].weight,s.gru.weight_ih_l0]:
                self.assertGreater(param.grad.abs().sum().item(),0)
            before=s(*inputs)[0]
            with torch.no_grad():
                for param in s.aux_head.parameters(): param.add_(100)
            torch.testing.assert_close(before,s(*inputs)[0])

    def test_labels_cannot_change_inference(self):
        c=config();s=run.make_student(c);pool=FakePool('throw',c,2,1)
        prev=np.zeros((2,20),np.float32)
        a=run.student_step(s,pool.obs,prev,None,'catch')[0]
        pool.obs['object_target'][:]=np.nan
        pool.obs['teacher_obs']['object']['pos3d'][:]=np.nan
        pool.obs['teacher_obs']['object']['v_lin_3d'][:]=np.nan
        pool.obs['task_id']=999
        np.testing.assert_array_equal(a,run.student_step(s,pool.obs,prev,None,'catch')[0])

    def test_mask_and_chunk_equivalence(self):
        c=config();pred=torch.randn(2,4,6,requires_grad=True);target=torch.ones_like(pred)
        valid=torch.tensor([[1.,1,1,1],[1.,1,0,0]]);mask=torch.ones_like(pred);mask[:,0,3:]=0
        mask[1,2:]=0;target[1,2:]=float('nan');lengths=valid.sum(1)
        whole=auxiliary_losses(pred,target,mask,valid,c['auxiliary'],lengths)
        parts=[auxiliary_losses(pred[:,i:i+2],target[:,i:i+2],mask[:,i:i+2],valid[:,i:i+2],c['auxiliary'],lengths) for i in (0,2)]
        for k in whole: torch.testing.assert_close(whole[k],sum(x[k] for x in parts))
        sum(whole.values()).backward()
        self.assertEqual(pred.grad[1,2:].abs().sum().item(),0)
        self.assertEqual(pred.grad[:,0,3:].abs().sum().item(),0)

    def test_train_eval_resume(self):
        for stage in ('track','catch'):
            c=config(stage);adapters=run.validate(c)
            teachers={t:FakeTeacher(8 if stage=='track' else 20) for t in c['tasks']}
            with tempfile.TemporaryDirectory() as d,patch.object(run,'Pool',FakePool),patch.object(run,'provenance',return_value={'test':'stable'}):
                output=Path(d);run.train(c,teachers,adapters,output)
                saved=torch.load(output/'student_last.pth',weights_only=True)
                self.assertEqual(saved['schema'],SCHEMA)
                s=run.make_student(c);s.load_state_dict(saved['student'])
                result=run.evaluate(c,adapters,student=s)
                self.assertTrue(all(v['success_rate']==1 for v in result.values()))
                c['iterations']=3;run.train(c,teachers,adapters,output,output/'resume_last.pth')
                saved=torch.load(output/'resume_last.pth',weights_only=True)
                self.assertEqual(saved['iteration'],3)
                rows=[json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
                self.assertTrue(all('position_loss' in row and 'velocity_loss' in row for row in rows))

    def test_worker_pairs_reset_depth_with_reset_state(self):
        captured = []
        class Environment:
            act_t_dim, obs_t_dim = 8, 18

            def __init__(self, **kwargs):
                self.episode = 0
                self.step_touch = False
                self.Dcmm = types.SimpleNamespace(data=types.SimpleNamespace(time=0.))

            def reset(self):
                self.episode += 1
                self.Dcmm.data.time = 0.
                self.render()  # Must be the disabled legacy renderer, not camera.
                obs = observation(1)
                obs = {k: {a: b[0] for a, b in v.items()} if isinstance(v, dict) else v[0]
                       for k, v in obs.items()}
                obs['object']['pos3d'][:] = self.episode
                return obs, {}

            def step(self, action):
                self.step_touch = True
                self.Dcmm.data.time += .04
                return None, 1., False, True, {'success': True}

            def _get_relative_object_pos3d(self):
                return np.full(3, self.episode, np.float32)

            def close(self):
                pass

        class Camera:
            def __init__(self, env, config):
                self.env = env
                self.name = config['camera']

            def capture(self):
                return np.full((1, 64, 64), self.env.episode / 10, np.float16)

            def capture_metres(self):
                captured.append((self.name, self.env.episode, self.env.Dcmm.data.time))
                return np.full((64, 64), self.env.episode / 10, np.float32)

            def close(self):
                pass

        class Pipe:
            def __init__(self):
                self.commands = iter([('step', {}), ('close', None)])
                self.messages = []

            def recv(self):
                return next(self.commands)

            def send(self, message):
                self.messages.append(message)

            def close(self):
                pass

        c = run.env_config(config('track'), 'throw')
        c['env_factory'] = 'fake_visual_worker:Environment'
        pipe = Pipe()
        module = types.ModuleType('fake_visual_worker')
        module.Environment = Environment
        with patch.dict('sys.modules', {'fake_visual_worker': module}), \
                patch.object(environments, 'DepthCamera', Camera), patch.dict('os.environ', {}):
            environments._worker(pipe, 'throw', c, 1, False)
        self.assertTrue(all(status == 'ok' for status, _ in pipe.messages))
        initial = pipe.messages[0][1]
        next_packet, _, done, info = pipe.messages[1][1]
        self.assertTrue(done)
        np.testing.assert_array_equal(next_packet['object_target'][:3], [2,2,2])
        np.testing.assert_array_equal(next_packet['object_valid'], [1,1,1,0,0,0])
        self.assertTrue(info['track_contact_success'])
        self.assertEqual(initial['teacher_obs']['object']['pos3d'][0], 1)
        self.assertEqual(next_packet['teacher_obs']['object']['pos3d'][0], 2)
        self.assertEqual(next_packet['depth'].shape, (2, 64, 64))
        self.assertAlmostEqual(float(next_packet['depth'][0, 0, 0]), (.2-.1)/4.9, places=3)
        self.assertAlmostEqual(float(next_packet['depth'][1, 0, 0]), (.2-.01)/1.49, places=3)
        self.assertNotIn('camera_depth_metres', next_packet)
        captured.clear()

        c['inspection_cameras'] = ['top', 'wrist']
        pipe = Pipe()
        with patch.dict('sys.modules', {'fake_visual_worker': module}), \
                patch.object(environments, 'DepthCamera', Camera), patch.dict('os.environ', {}):
            environments._worker(pipe, 'throw', c, 1, False)
        self.assertTrue(all(status == 'ok' for status, _ in pipe.messages))
        self.assertEqual(captured, [('top', 1, 0.), ('wrist', 1, 0.), ('top', 2, 0.), ('wrist', 2, 0.)])
        packet = pipe.messages[1][1][0]
        self.assertEqual(packet['episode_id'], 2)
        self.assertAlmostEqual(float(packet['camera_depth_metres']['wrist'][0, 0]), .2)


if __name__=='__main__': unittest.main()
