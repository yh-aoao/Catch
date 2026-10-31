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
register(id="gym_dcmm/BasketObjectTest-v0", entry_point="gym_dcmm.envs.basket_object_test:make_basket_object_test")
register(id="gym_dcmm/RollAdaptiveWorld-v0", entry_point="gym_dcmm.envs.roll_adaptive:make_roll_adaptive")
register(id="gym_dcmm/RollPhysicalWorld-v1", entry_point="gym_dcmm.envs.roll_physical:make_roll_physical")
register(id="gym_dcmm/BounceStableWorld-v0", entry_point="gym_dcmm.envs.bounce_stable:make_bounce_stable")
register(id="gym_dcmm/BouncePhysicalWorld-v2", entry_point="gym_dcmm.envs.bounce_physical_v2:make_bounce_physical_v2")
