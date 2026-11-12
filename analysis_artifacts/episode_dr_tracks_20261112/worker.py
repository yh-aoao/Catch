"""Use production Track agents and the exact-quota episode DR evaluation loop."""
from pathlib import Path
import sys
ROOT=Path('/home/isee/Catch/catch_it_copy');sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from _eval_multitarget_worker import evaluate
from gym_dcmm.algs.ppo_dcmm_645edc4.ppo_dcmm_track import PPO_Track as Roll
from gym_dcmm.algs.ppo_dcmm_july22.ppo_dcmm_track import PPO_Track as Bounce
Roll.test=evaluate;Bounce.test=evaluate
if __name__=='__main__':
    from train_DCMM import main
    main()
