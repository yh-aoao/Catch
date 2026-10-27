from gymnasium.envs.registration import register

register(
    id="gym_dcmm/SlideWorld-v0",
    entry_point="motion_slide.environment:SlideEnv",
)

register(
    id="gym_dcmm/DcmmVecWorld-v0",
    entry_point="gym_dcmm.envs:DcmmVecEnv",
    #  max_episode_steps=300,
)

register(id="gym_dcmm/ObjectTest-v0", entry_point="gym_dcmm.envs.object_test:make_object_test")
register(id="gym_dcmm/RollAdaptiveWorld-v0", entry_point="gym_dcmm.envs.roll_adaptive:make_roll_adaptive")
