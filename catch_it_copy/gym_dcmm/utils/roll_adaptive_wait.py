"""Versioned reward-only Roll interception shaping, in world coordinates."""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AdaptiveWaitConfig:
    min_height: float = .38
    max_height: float = .62
    height_band: float = .06
    stand_off: float = .10
    speed_lookahead: float = .08
    hand_margin: float = .03
    xy_weight: float = 1.
    xy_scale: float = .30
    progress_weight: float = 5.
    height_weight: float = 1.5
    hand_weight: float = 4.
    ready_weight: float = .5
    settle_speed: float = .15
    early_contact_weight: float = 2.

    def __post_init__(self):
        if any(not math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('Adaptive waiting parameters must be finite')
        if not 0 < self.min_height < self.max_height:
            raise ValueError('Adaptive waiting requires 0 < min_height < max_height')
        if min(self.xy_scale, self.settle_speed, self.height_band) <= 0:
            raise ValueError('Adaptive waiting scales must be positive')
        if any(v < 0 for k, v in asdict(self).items() if k not in ('min_height', 'max_height')):
            raise ValueError('Adaptive waiting weights and distances must be nonnegative')


def interception(position, velocity, radius, edge, table_top, gravity, cfg):
    """Predict front-edge exit, then choose a point on the free-flight trajectory.

    The waiting height follows the chosen fall time, bounded by a broad robot
    workspace band; it is never independently clamped off the predicted path.
    The prediction assumes constant horizontal velocity until exit, not a
    calibrated model of the future rolling deceleration.
    """
    p, v = np.asarray(position, float), np.asarray(velocity, float)
    top = table_top + radius
    if cfg.max_height >= top:
        raise ValueError('Maximum capture height must be below the ball center on the table')
    g = max(abs(float(gravity)), 1e-6)
    forward = max(0., -float(v[1]))
    t_min = math.sqrt(2 * (top - cfg.max_height) / g)
    t_max = math.sqrt(2 * (top - cfg.min_height) / g)
    distance = cfg.stand_off + forward * cfg.speed_lookahead
    fall_time = float(np.clip(distance / max(forward, .01), t_min, t_max))
    exit_time = max(0., (p[1] - edge) / forward) if forward > .01 else 0.
    exit_position = p[:2] + v[:2] * exit_time
    # Without forward motion there is no meaningful predicted exit. Keep an
    # outside-table target rather than moving the hand under a stationary ball.
    if forward <= .01:
        exit_position = np.array([p[0], edge])
    goal = np.array([exit_position[0] + v[0]*fall_time,
                     edge + v[1]*fall_time if forward > .01 else edge-cfg.stand_off,
                     top - .5*g*fall_time**2])
    return goal, dict(exit_time=float(exit_time), fall_time=fall_time,
                      forward_speed=forward, predicted_exit_x=float(exit_position[0]))


def terms(palm, previous_palm, palm_velocity, goal, waiting, hand_front, edge,
          hand_contact, table_contact, cfg):
    palm, goal = np.asarray(palm), np.asarray(goal)
    xy_error = float(np.linalg.norm(palm[:2] - goal[:2]))
    dz = abs(float(palm[2] - goal[2]))
    intrusion = max(0., hand_front + cfg.hand_margin - edge) if waiting else 0.
    # Reward proximity/settling only when the full hand has front clearance.
    clearance_gate = float(np.clip(1. - intrusion/.10, 0., 1.)) if waiting else 1.
    progress = 0.
    if previous_palm is not None:
        axes = slice(0, 2) if waiting else slice(0, 3)
        # Both distances use today's same goal: target drift earns no progress.
        progress = cfg.progress_weight * (np.linalg.norm(np.asarray(previous_palm)[axes]-goal[axes])
                                           - np.linalg.norm(palm[axes]-goal[axes]))
    proximity = cfg.xy_weight / (1 + (xy_error/cfg.xy_scale)**2)
    excess_height = max(0., dz-cfg.height_band)
    height = (-cfg.height_weight * min(1., excess_height/.15) if waiting else
              .4/(1+(dz/.10)**2))
    ready = (cfg.ready_weight * math.exp(-(xy_error/.12)**2 - (excess_height/.10)**2)
             * math.exp(-float(np.dot(palm_velocity, palm_velocity))/cfg.settle_speed**2)
             * clearance_gate if waiting else 0.)
    result = dict(xy=float(proximity*clearance_gate), progress=float(progress), height=float(height),
                  hand_clearance=-cfg.hand_weight*min(1., intrusion/.15), ready=float(ready),
                  early_contact=-cfg.early_contact_weight*float(waiting and hand_contact and table_contact))
    return result
