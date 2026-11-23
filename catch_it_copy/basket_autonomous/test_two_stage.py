"""Two-stage curriculum checks; inference/backward only, no training run."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .environment import AutonomousTeacherEnv
from .policy import Policy
from .run import package,validate_initialization,evaluation_options

class CurriculumTests(unittest.TestCase):
    def config(self,stage):
        cfg=json.loads((Path(__file__).parent/'configs'/f'{stage}_stage.json').read_text())
        cfg['options']['pose_bank_entries']=json.loads((Path(__file__).parent/'configs/caught_pose_bank.json').read_text())
        cfg['options'].pop('pose_bank',None)
        return cfg

    def test_import_and_validation(self):
        torch.set_num_threads(1);model=Policy();source=package(model,self.config('fixed'),123)
        target=self.config('mobile');validate_initialization(source,target)
        other=Policy();other.load_state_dict(source['model'],strict=True)
        x=torch.randn(3,65);torch.testing.assert_close(model.act(x),other.act(x))
        self.assertTrue(all(p.requires_grad for p in other.parameters()))
        self.assertEqual(evaluation_options(target)['fixed_replay_probability'],0.)
        bad=copy.deepcopy(source);bad['config']['options']['training_stage']='joint'
        with self.assertRaises(ValueError):validate_initialization(bad,target)
        bad=copy.deepcopy(target);bad['options']['hand_jitter_rad']=0.
        with self.assertRaises(ValueError):validate_initialization(source,bad)

    def test_fixed_masks_only_base_probability(self):
        torch.set_num_threads(1);model=Policy();x=torch.randn(4,65)
        mask=torch.ones(4,20);mask[:,:2]=0.
        _,raw,logp,_=model.sample(x,mask)
        model.zero_grad();(-model.log_probability(model.distribution(x),raw.detach(),mask).mean()).backward()
        self.assertEqual(float(model.actor[-1].weight.grad[:2].abs().sum()),0.)
        self.assertGreater(float(model.actor[-1].weight.grad[2:].abs().sum()),0.)
        model.zero_grad();(-model.log_probability(model.distribution(x),raw.detach()).mean()).backward()
        self.assertGreater(float(model.actor[-1].weight.grad[:2].abs().sum()),0.)

    def test_fixed_and_mobile_physics_no_assists(self):
        for stage,probability,fixed in [('fixed',0.,True),('mobile',0.,False),('mobile',1.,True)]:
            cfg=self.config(stage);cfg['options']['fixed_replay_probability']=probability
            env=AutonomousTeacherEnv(cfg['options'])
            try:
                obs,info=env.reset(seed=123)
                self.assertEqual(info['fixed_episode'],fixed);self.assertTrue(env.touching())
                before_arm=env.Dcmm.target_arm_qpos.copy();before_hand=env.Dcmm.target_hand_qpos.copy()
                command=np.r_[.2,.1,np.full(6,.2),np.full(12,.1)].astype(np.float32)
                # Release-readiness state must never trigger actuator braking.
                env.basket_mobile_state.update(unlocked=True,ready_time=0.)
                with patch.object(env.Dcmm,'set_throw_pos_vel',side_effect=AssertionError('teleport')),patch.object(env,'_basket_parking_state',side_effect=AssertionError('waypoint')):
                    obs,reward,done,trunc,info=env.step(command)
                speed=np.linalg.norm(env.Dcmm.target_base_vel[:2])
                if fixed:self.assertEqual(speed,0.)
                else:self.assertGreater(speed,0.)
                self.assertGreater(np.linalg.norm(env.Dcmm.target_arm_qpos-before_arm),0.)
                self.assertGreater(np.linalg.norm(env.Dcmm.target_hand_qpos-before_hand),0.)
                self.assertEqual(float(env.Dcmm.data.ctrl[-1]),0.)
                self.assertTrue(np.all(env.Dcmm.data.qfrc_applied[36:42]==0.))
                self.assertTrue(np.all(env.Dcmm.data.xfrc_applied[env.Dcmm.model.body('object').id]==0.))
                self.assertTrue(np.isfinite(obs).all());self.assertTrue(np.isfinite(reward))
            finally:env.close()

if __name__=='__main__':unittest.main()
