"""Instrument production restore, then run an actual short PPO training job."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from gym_dcmm.algs.ppo_dcmm_5fe75d5f.ppo_dcmm_catch_hold import PPO_Catch_Hold

def wrap(original):
    def checked(self,path):
        steps=self.agent_steps;epoch=self.epoch_num;best=self.best_rewards
        original(self,path)
        assert self.catch_warm_start
        raw=torch.load(path,map_location=self.device)
        for name in ('model','running_mean_std_track','running_mean_std_hand','value_mean_std'):
            actual=getattr(self,name).state_dict()
            assert set(actual)==set(raw[name])
            for key in actual:assert torch.equal(actual[key],raw[name][key]),(name,key)
        assert (self.agent_steps,self.epoch_num,self.best_rewards)==(steps,epoch,best)
        assert len(self.optimizer.state)==0,'Old optimizer was restored'
        print('[warm-start-validation] exact model/stats; fresh optimizer, counters and best score',flush=True)
    return checked

PPO_Catch_TwoStage.restore_train=wrap(PPO_Catch_TwoStage.restore_train)
PPO_Catch_Hold.restore_train=wrap(PPO_Catch_Hold.restore_train)
from train_DCMM import main
if __name__=='__main__':
    main()
