"""Optional evaluation-time command limiter before the ball leaves the table.

This projects commanded motion, not measured robot state: tracking error and
control delay mean it is not a swept-volume or hardware collision guarantee.
"""
import numpy as np

from gym_dcmm.utils.roll_rewards import hand_collision_ids


class RollWaitGuard:
    def __init__(self, env):
        self.env = env
        self.released = False
        self.diagnostics = dict(active=False, intervention=False)

    def filter(self, action):
        env = self.env
        robot = env.Dcmm
        model, data = robot.model, robot.data
        table = model.geom('table_surface').id
        edge = float(data.geom_xpos[table, 1] - model.geom_size[table, 1])
        radius = float(model.geom_size[env.object_id, 0])
        position = data.body('object').xpos
        object_body = model.body('object').id
        object_ids = set(np.flatnonzero(model.geom_bodyid == object_body).tolist())
        table_contact = any((int(c.geom1) == table and int(c.geom2) in object_ids) or
                            (int(c.geom2) == table and int(c.geom1) in object_ids)
                            for c in data.contact)
        if position[1] + radius < edge and not table_contact:
            self.released = True
        self.diagnostics = dict(active=not self.released, intervention=False, edge_y=edge,
                                ball_clear=bool(self.released), table_contact=bool(table_contact))
        if self.released:
            return action

        result = {key: np.asarray(value, dtype=float).copy() for key, value in action.items()}
        hand_ids = hand_collision_ids(model, env.hand_start_id)
        hand_front = float(np.max(data.geom_xpos[hand_ids, 1] + model.geom_rbound[hand_ids]))
        palm = (data.geom_xpos[env.hand_start_id] +
                data.body('link6').xmat.reshape(3, 3)[:, 1] * radius)
        wrist = data.body('link6').xpos
        world_base = data.body('arm_base')
        local_base = robot.data_arm.body('arm_base')
        rotation = world_base.xmat.reshape(3, 3) @ local_base.xmat.reshape(3, 3).T
        internal_wrist = (world_base.xpos + rotation @
                          (robot.data_arm.body('link6').xpos - local_base.xpos))
        candidate = internal_wrist + rotation @ result['arm'][:3]
        # Include the measured full hand's conservative forward extent.
        forward_offset = hand_front - wrist[1]
        candidate[1] = min(candidate[1], edge - .05 - forward_offset)
        # Hold the existing training reward's 0.50m capture-point height.
        candidate[2] = .50 - (palm[2] - wrist[2])
        delta = rotation.T @ (candidate - internal_wrist)
        result['arm'][:3] = np.clip(delta, -.025, .025)
        # Keep the current target orientation and open-hand targets while waiting.
        result['arm'][3:] = 0.
        result['hand'][:] = 0.

        dt = float(env.steps_per_policy * model.opt.timestep)
        base_rotation = data.body('base_link').xmat.reshape(3, 3)[:2, :2]
        world_velocity = base_rotation @ result['base'][:2]
        arm_dy = float((rotation @ result['arm'][:3])[1])
        # Begin braking 15cm before the conservative hand boundary, accounting
        # for the arm command during this policy interval.
        allowed = max(0., (edge - .15 - hand_front - max(0., arm_dy)) / dt)
        world_velocity[1] = min(world_velocity[1], allowed)
        result['base'][:2] = np.linalg.solve(base_rotation, world_velocity)
        self.diagnostics.update(intervention=any(not np.allclose(result[k], action[k]) for k in result),
                                hand_front_y=hand_front, palm_z=float(palm[2]),
                                commanded_wait_z=.50)
        return result
