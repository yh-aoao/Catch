"""Basket parking geometry and rewards, independent of the simulator."""
import numpy as np


def limit_speed(velocity, maximum):
    velocity = np.asarray(velocity, dtype=float)
    return velocity * min(1.0, maximum / max(float(np.linalg.norm(velocity)), 1e-9))


def parking_state(base_position, base_velocity, yaw, basket_center, cfg):
    target = np.asarray(basket_center, dtype=float).copy()
    target[1] -= cfg.basket_base_front_dist
    delta = target - base_position
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, s], [-s, c]])
    error = rotation @ delta[:2]
    velocity = rotation @ np.asarray(base_velocity)[:2]
    in_position = bool(np.all(np.abs(delta[:2]) < cfg.basket_track_position_tolerance))
    desired = (np.zeros(2) if in_position else
               limit_speed(cfg.basket_track_slowdown_gain * error, cfg.basket_track_max_speed))
    return dict(target=target, error_world=delta[:2],
                observation=np.r_[error, delta[2]], velocity=velocity,
                desired_velocity=desired, distance=float(np.linalg.norm(error)),
                speed=float(np.linalg.norm(velocity)), in_position=in_position,
                settled=in_position and np.linalg.norm(velocity) < cfg.basket_track_speed_tolerance)


def parking_reward(state, previous_distance, action, success, failed, cfg):
    # Signed progress: moving away must undo the credit for approaching.
    terms = dict(
        distance=-cfg.basket_track_w_distance * state['distance'],
        progress=cfg.basket_track_w_progress * (previous_distance - state['distance']),
        velocity=-cfg.basket_track_w_velocity * float(np.sum(
            (state['velocity'] - state['desired_velocity']) ** 2)),
        control=-cfg.basket_w_ctrl_base * float(np.sum(np.asarray(action) ** 2)),
        time=-cfg.basket_track_time_cost,
        success=cfg.basket_track_success_reward if success else 0.0,
        failure=-cfg.basket_track_failure_cost if failed else 0.0,
    )
    return float(sum(terms.values())), terms
