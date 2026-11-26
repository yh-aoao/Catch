import copy,json,pickle,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .environment import AutonomousTeacherEnv
from .state import capture,restore,VERSION
from .policy import Policy
from .run import package,validate_initialization,evaluation_options,evaluate
from .pool import Pool
ROOT=Path(__file__).resolve().parents[1]
def config(name):
    c=json.loads((ROOT/'basket_prepare_throw/configs'/name).read_text());o=c['options']
    o['pose_bank_entries']=json.loads((ROOT/o.pop('pose_bank')).read_text());return c
class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.c=config('stage1_prepare.json')
    def test_real_pose_bank_and_prepare(self):
        self.assertEqual(len(self.c['options']['pose_bank_entries']),20)
        e=AutonomousTeacherEnv(dict(self.c['options'],episode_seconds=.04))
        try:
            obs,_=e.reset(seed=123);self.assertEqual(obs.shape,(66,));self.assertEqual(obs[-1],1.)
            _,r,done,trunc,info=e.step(np.zeros(20));self.assertTrue(done or trunc)
            self.assertEqual(info['episode_kind'],'prepare');self.assertTrue(np.isfinite(r))
            self.assertNotIn('release',info['reward_terms']);self.assertEqual(info['reward_terms']['failure'],-20.)
        finally:e.close()
    def test_physical_replay_and_continuation(self):
        e=AutonomousTeacherEnv(self.c['options'])
        try:
            e.reset(seed=123);e.step(np.zeros(20));state=capture(e)
            target=e.Dcmm.target_arm_qpos.copy();q=e.Dcmm.data.qpos.copy();v=e.Dcmm.data.qvel.copy()
            action=np.full(20,.01);rng_state=np.random.get_state();e.step(action);expected=e.Dcmm.data.qpos.copy()
            restore(e,state)
            np.testing.assert_array_equal(e.Dcmm.data.qpos,q);np.testing.assert_array_equal(e.Dcmm.data.qvel,v)
            np.testing.assert_array_equal(e.Dcmm.target_arm_qpos,target)
            np.testing.assert_allclose(e.observation(),state['observation'],atol=1e-6)
            np.random.set_state(rng_state)
            e.step(action);np.testing.assert_allclose(e.Dcmm.data.qpos,expected,atol=1e-9)
        finally:e.close()
    def test_stage_two_bank_and_import(self):
        first=copy.deepcopy(self.c);first['options']['preparation_level']=2
        second=config('stage2_throw.json');m=Policy();saved=package(m,first,0);validate_initialization(saved,second)
        other=Policy();other.load_state_dict(saved['model']);self.assertTrue(all(p.requires_grad for p in other.parameters()))
        e=AutonomousTeacherEnv(first['options'])
        try:e.reset(seed=123);snapshot=capture(e)
        finally:e.close()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bank.pkl';path.write_bytes(pickle.dumps(dict(version=VERSION,states=[snapshot],source_options=first['options'])))
            e=AutonomousTeacherEnv(dict(second['options'],ready_bank=str(path),evaluation_task='short_throw'))
            try:
                obs,_=e.reset(seed=2);self.assertEqual(obs[-1],0.)
                np.testing.assert_array_equal(e.Dcmm.target_arm_qpos,snapshot['dcmm']['target_arm_qpos'])
                e.step(np.zeros(20))
            finally:e.close()
    def test_prepare_success_and_full_handoff(self):
        for stage,kind in [('prepare','prepare'),('throw','full')]:
            e=AutonomousTeacherEnv(dict(self.c['options'],two_stage=stage,evaluation_task=kind))
            try:
                e.reset(seed=123);q=e.Dcmm.data.qpos.copy()
                metrics=dict(ready=True,posture_quality=1.)
                info=dict(reward_terms=dict(time=-.02,score=0.,failure=0.),hand_contact=True,safe_basket_success=False)
                with patch('basket_autonomous.environment.AutonomousTeacherEnv.step',return_value=(e.observation(),0.,False,False,info)),patch('basket_prepare_throw.environment.preparation_metrics',return_value=metrics):
                    for _ in range(8):result=e.step(np.zeros(20))
                    self.assertTrue(result[4]['prepare_success']);self.assertIsNotNone(e.last_ready_snapshot)
                    self.assertEqual(result[2],stage=='prepare')
                    if stage=='throw':self.assertEqual(result[0][-1],0.)
                    np.testing.assert_array_equal(e.Dcmm.data.qpos,q)
            finally:e.close()
    def test_spawn_and_evaluation(self):
        options=dict(self.c['options'],episode_seconds=.04)
        pool=Pool(2,options,123)
        try:
            self.assertEqual(pool.obs.shape,(2,66));result=pool.step(np.zeros((2,20)))
            self.assertTrue(np.isfinite(result[1]).all())
        finally:pool.close()
        c=copy.deepcopy(self.c);c['options']['episode_seconds']=.04
        result=evaluate(Policy(),c,2,123)
        self.assertEqual(result['episodes'],2);self.assertEqual(result['successes'],0)
if __name__=='__main__':unittest.main()
