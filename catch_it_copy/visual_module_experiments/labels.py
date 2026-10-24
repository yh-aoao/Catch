"""Simulation-only labels. Never passed to the student forward interface."""
import numpy as np
from gym_dcmm.utils.roll_rewards import hand_collision_ids


def hand_contact(env):
    model, data = env.Dcmm.model, env.Dcmm.data
    obj = int(model.geom('object').id)
    hand = set(hand_collision_ids(model, env.hand_start_id).tolist())
    return any((int(c.geom1) == obj and int(c.geom2) in hand) or
               (int(c.geom2) == obj and int(c.geom1) in hand) for c in data.contact)


def labels(env, sensor, specs, segmentation=False):
    model,data = env.Dcmm.model,env.Dcmm.data
    obj = int(model.geom('object').id)
    result = {'world_position': np.asarray(data.geom_xpos[obj],np.float32).copy(),
              'visible': np.zeros(len(specs),np.float32),
              'uv': np.zeros((len(specs),2),np.float32)}
    if not segmentation:
        return result
    renderer = sensor.renderer
    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    try:
        for i,spec in enumerate(specs):
            renderer.update_scene(data,camera=spec['name'])
            segmentation_image = np.asarray(renderer.render())
            mask = ((segmentation_image[...,0] == obj) &
                    (segmentation_image[...,1] == int(sensor.mujoco.mjtObj.mjOBJ_GEOM)))
            y,x = np.nonzero(mask)
            if len(x):
                result['visible'][i] = 1
                result['uv'][i] = [2*x.mean()/max(mask.shape[1]-1,1)-1,
                                   2*y.mean()/max(mask.shape[0]-1,1)-1]
    finally:
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
    return result
