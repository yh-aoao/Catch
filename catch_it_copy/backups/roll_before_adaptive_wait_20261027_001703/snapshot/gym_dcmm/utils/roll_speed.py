"""Validation and initial rolling velocity override for object evaluation."""
import math
from numbers import Real

import numpy as np


def speed_range(value):
    if value is None:
        return None
    values = [value, value] if isinstance(value, Real) else value
    if (not isinstance(values, (list, tuple)) or len(values) != 2 or
            any(isinstance(v, bool) or not isinstance(v, Real) or
                not math.isfinite(float(v)) or float(v) < 0 for v in values)):
        raise ValueError('roll_speed must be a nonnegative speed or [minimum, maximum] in m/s')
    low, high = map(float, values)
    if low > high:
        raise ValueError('roll_speed minimum must not exceed maximum')
    return low, high


def override_velocity(env, bounds):
    low, high = bounds
    speed = low if low == high else np.random.uniform(low, high)
    velocity = env.object_vel6d.copy()
    heading = math.atan2(float(velocity[0]), -float(velocity[1]))
    velocity[:3] = [speed * math.sin(heading), -speed * math.cos(heading), 0.]
    radius = max(float(env.Dcmm.model.geom_size[env.object_id][0]), 1e-4)
    velocity[3:] = [-velocity[1] / radius, velocity[0] / radius, 0.]
    env.object_vel6d = velocity
