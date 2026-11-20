"""Exercise scratch -> single-tennis free stopping without optimizer updates."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from gym_dcmm.envs.basket_mixed import BasketMixedEnv
from gym_dcmm.envs.basket_free_stop import BasketFreeStopEnv
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mixed import PPO_Basket_Mixed
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_basket_mobile_joint import PPO_Basket_MobileJoint


class TennisFreeStopTests(unittest.TestCase):
    def config(self,name):
        OmegaConf.register_new_resolver('resolve_default',lambda default,arg:default if arg=='' else arg,replace=True)
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base='1.1'):
            c=compose(config_name=name)
        c.rl_device='cpu';c.num_envs=1;c.train.ppo.num_actors=1;c.train.ppo.minibatch_size=64
        return c

    def vector(self,c,mobile):
        keys=['object_motion','basket_fixed_base_training','basket_control_probe','basket_arm_control',
            'basket_reward_version','basket_fixed_physical','basket_carry_reward','basket_carry_options',
            'basket_finger_reward','basket_finger_options','basket_fixed_hand_flex','basket_delivery_version',
            'basket_delivery_options','basket_mixed_ball','basket_mixed_options','basket_range']
        if mobile:keys+=['basket_free_stop']
        args={k:OmegaConf.to_container(c[k],resolve=True) if OmegaConf.is_config(c[k]) else c[k] for k in keys}
        args.update(task='Catching',render_mode=None,viewer=False,imshow_cam=False,basket_log=False)
        if mobile:args['basket_mobile_options']=OmegaConf.to_container(c.basket_mobile,resolve=True)
        cls=BasketFreeStopEnv if mobile else BasketMixedEnv
        return gym.vector.SyncVectorEnv([lambda:cls(**args)])

    def test_scratch_single_tennis_then_free_stop_transfer_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            fixed=self.config('basket_tennis_range')
            for key in ['basket_mixed_source','checkpoint_catching','checkpoint_tracking','basket_warm_start','basket_throw_checkpoint']:
                self.assertFalse(fixed[key])
            first_env=self.vector(fixed,False)
            self.addCleanup(first_env.close)
            with patch('torch.load',side_effect=AssertionError('Scratch must not load weights')):
                first=PPO_Basket_Mixed(first_env,folder+'/stage1',fixed)
            self.addCleanup(first.writer.close)
            first.set_eval()
            self.assertEqual(first.source_metadata,{'kind':'scratch'})
            self.assertIsNone(first.reference)
            self.assertFalse(first.input_normalization_frozen)
            for seed in (123,127,131):
                first_env.reset(seed=seed)
                self.assertEqual(first_env.envs[0].unwrapped.ball_name,'tennis_ball')
            first.save(folder+'/fixed_tennis')
            mobile=self.config('basket_mobile_tennis_free_stop')
            mobile.basket_throw_checkpoint=folder+'/fixed_tennis.pth'
            second_env=self.vector(mobile,True)
            self.addCleanup(second_env.close)
            second=PPO_Basket_MobileJoint(second_env,folder+'/stage2',mobile)
            self.addCleanup(second.writer.close)
            second.set_eval()
            self.assertEqual(second.obs_shape,(51,))
            self.assertFalse(second.conditioned)
            self.assertEqual(second.single_ball,'tennis_ball')
            self.assertTrue(all(p.requires_grad for p in second.model.parameters()))
            raw=torch.randn(8,51);raw[:,47]=1.
            fixed_raw=raw[:,:35].clone();fixed_raw[:,34]+=.2
            profile=second_env.envs[0].unwrapped.profiles['tennis_ball']
            properties=torch.tensor([[profile['radius_m'],profile['mass_kg']]]).expand(8,2)
            expected=first.model.act_inference(dict(obs=first.running_mean_std(torch.cat((fixed_raw,properties),-1))))
            actual=second.model.act_inference(dict(obs=second.running_mean_std(raw)))[:,2:]
            torch.testing.assert_close(expected,actual,atol=2e-6,rtol=2e-5)
            for seed in (123,127,131):
                obs,_=second_env.reset(seed=seed)
                env=second_env.envs[0].unwrapped
                self.assertEqual(env.ball_name,'tennis_ball')
                for _ in range(3):
                    second.obs={'obs':second.obs2tensor(obs)}
                    with torch.no_grad():
                        action=second.model.act_inference(dict(obs=second.running_mean_std(second.obs['obs']))).clamp(-1,1)
                    obs,reward,_,_,_=second_env.step(second.action2dict(action))
                    self.assertTrue(np.isfinite(reward).all())
            second.save(folder+'/mobile_tennis')
            mobile.test=True
            loaded=PPO_Basket_MobileJoint(second_env,folder+'/reload',mobile)
            self.addCleanup(loaded.writer.close)
            loaded.restore_test(folder+'/mobile_tennis.pth')
            for k,v in second.model.state_dict().items():
                torch.testing.assert_close(v,loaded.model.state_dict()[k])

if __name__=='__main__':unittest.main()
