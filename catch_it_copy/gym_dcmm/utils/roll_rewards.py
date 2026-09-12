"""Roll interception targets and hand shaping in explicit world/joint coordinates."""
import numpy as np


def interception_target(position, velocity, radius, cfg, gravity=9.81, wait_height=None):
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    table_min = cfg.roll_table_pos[:2] - cfg.roll_table_size[:2]
    table_max = cfg.roll_table_pos[:2] + cfg.roll_table_size[:2]
    on_table = bool(np.all(position[:2] >= table_min) and
                    np.all(position[:2] <= table_max) and
                    position[2] >= cfg.roll_table_height - 0.01)
    target = position.copy()
    wait_z = cfg.roll_wait_height if wait_height is None else wait_height
    ball_plane = wait_z + cfg.roll_intercept_ball_offset
    if on_table:
        # Predict the front-edge exit, then the fall to the interception height.
        edge_time = (max(0., (table_min[1] - position[1]) / velocity[1])
                     if velocity[1] < -0.01 else 0.)
        fall_time = np.sqrt(2. * max(0., cfg.roll_table_height + radius - ball_plane) / gravity)
        target[:2] += velocity[:2] * (edge_time + fall_time)
        # Keep the physically predicted Y even for slow balls. Clearance is a
        # full-hand 3D constraint; clamping Y can make the ball miss this target.
        target[2] = wait_z
    elif position[2] > ball_plane:
        # The ball has left the table but is still above the catching plane.
        fall_time = (velocity[2] + np.sqrt(velocity[2] ** 2 +
                     2. * gravity * (position[2] - ball_plane))) / gravity
        target[:2] += velocity[:2] * fall_time
        target[2] = wait_z
    else:
        target[2] -= cfg.roll_intercept_ball_offset
    return target, on_table


def hand_workspace(centers, radii, ee_position, cfg):
    """Conservative sphere/table-box separation, not swept-volume collision checking."""
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if len(centers) == 0 or len(centers) != len(radii):
        raise ValueError('Roll workspace requires collision-enabled hand geometry')
    # Signed box distance also handles spheres whose centers are inside the slab.
    delta = np.abs(centers - cfg.roll_table_pos) - cfg.roll_table_size
    distance = np.linalg.norm(np.maximum(delta, 0.), axis=1) + np.minimum(
        np.max(delta, axis=1), 0.) - radii
    clearance = float(np.min(distance))
    top_offset = float(np.max(centers[:, 2] + radii) - ee_position[2])
    ceiling = cfg.roll_table_pos[2] - cfg.roll_table_size[2] - cfg.roll_grasp_clearance - top_offset
    wait_height = float(np.clip(ceiling, cfg.roll_min_wait_height, cfg.roll_wait_height))
    return dict(clearance=clearance, hand_clear=clearance >= cfg.roll_grasp_clearance,
                wait_height=wait_height, height_feasible=ceiling >= cfg.roll_min_wait_height)


def capture_ready(position, radius, table_contact, workspace, cfg):
    edge = cfg.roll_table_pos[1] - cfg.roll_table_size[1]
    # Require the whole ball to clear the front edge, not just its center.
    ball_clear = bool(position[1] + radius < edge and not table_contact)
    return ball_clear, bool(ball_clear and workspace['hand_clear'])


def wait_target(candidate, locked_target, ee_position, on_table, cfg):
    if locked_target is not None:
        return np.asarray(locked_target).copy(), np.asarray(locked_target).copy()
    candidate = np.asarray(candidate).copy()
    lock = bool(on_table and np.linalg.norm(candidate - ee_position) < cfg.roll_wait_lock_distance)
    return candidate, candidate.copy() if lock else None


def position_terms(ee_position, ee_velocity, target, on_table, previous_distance, cfg):
    ee = np.asarray(ee_position)
    delta = np.asarray(target) - ee
    distance = float(np.linalg.norm(delta))
    xy = float(np.linalg.norm(delta[:2]))
    terms = dict(
        xy=cfg.roll_w_xy / (1. + (xy / cfg.roll_sigma_xy) ** 2),
        height=cfg.roll_w_h / (1. + (delta[2] / cfg.roll_sigma_h) ** 2),
        approach=cfg.roll_w_approach * (0. if previous_distance is None else
                                      previous_distance - distance),
        # Waiting near the target should not require moving continuously.
        waiting=-cfg.roll_w_wait_speed * float(np.sum(np.asarray(ee_velocity) ** 2))
                * np.exp(-(distance / .15) ** 2) if on_table else 0.,
        above_wait=-cfg.roll_w_above_wait * max(0., ee[2] - cfg.roll_wait_height)
                   if on_table else 0.,
    )
    return terms, distance


def hand_terms(qpos, distance, contact, cfg, allow_closure=True):
    # The simulator has 16 joints; policy observations contain only 12.
    qpos = np.asarray(qpos, dtype=float).reshape(16)
    fingers = qpos[[0, 2, 3, 4, 6, 7, 8, 10, 11]].reshape(3, 3)
    thumb = qpos[[13, 14, 15]]
    flex = np.r_[fingers.ravel(), thumb]
    gate = 1. if contact else float(np.clip(1. - distance / cfg.roll_hand_close_distance, 0., 1.))
    if not allow_closure:
        gate = 0.
    ready = np.asarray(cfg.roll_hand_ready_target)
    closed = np.asarray(cfg.roll_hand_close_target)
    target = ready + gate * (closed - ready)
    pose_error = float(np.mean((fingers - target) ** 2))
    pose_error += .25 * float(np.mean((thumb - cfg.roll_hand_thumb_target) ** 2))
    # Bounded closure credit near the ball, no incentive to squeeze to joint limits.
    return dict(
        hand_pose=-cfg.roll_w_hand_pose * pose_error,
        hand_closure=cfg.roll_w_hand_closure * gate * np.exp(-pose_error / .16),
        hand_sync=-cfg.roll_w_hand_sync * float(np.mean(np.var(fingers, axis=0))),
        hand_chain=-cfg.roll_w_hand_chain * float(np.mean(
            (fingers[:, 1:] - fingers[:, :1] * np.array([.75, .65])) ** 2)),
        hand_limits=-cfg.roll_w_hand_limits * float(np.mean(
            np.minimum(flex, 0.) ** 2 + np.maximum(flex - cfg.roll_hand_flex_max, 0.) ** 2)),
    )
