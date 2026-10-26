"""Fixed-base near-target reward; independent from launch-only defaults."""
from types import SimpleNamespace
import numpy as np
from gym_dcmm.utils.basket_simple import reward as launch_reward


def make_config(base):
    cfg = SimpleNamespace(**{k: getattr(base, k) for k in dir(base) if not k.startswith('__')})
    cfg.basket_center = np.array([0., 1.05, .55])
    cfg.basket_simple_progress_bonus = 5.
    cfg.basket_simple_collision_cost = 40.
    cfg.basket_min_release_distance = .3
    cfg.basket_min_flight_time = .1
    cfg.basket_min_flight_travel = .2
    cfg.basket_aim_sigma = .25
    cfg.basket_aim_bonus = 10.
    cfg.basket_aim_score_bonus = 20.
    cfg.basket_aim_follow_seconds = .3
    return cfg


def crossing_error(position, velocity, center, gravity, tilt_deg):
    normal = np.array([0., -np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))])
    p, v, g, c = map(np.asarray, (position, velocity, gravity, center))
    a, b, d = .5*float(g@normal), float(v@normal), float((p-c)@normal)
    roots = np.roots([a,b,d]) if abs(a) > 1e-10 else ([-d/b] if abs(b) > 1e-10 else [])
    valid = [float(np.real(t)) for t in roots if abs(np.imag(t)) < 1e-8
             and 0 < float(np.real(t)) <= 2. and b+2*a*float(np.real(t)) < 0]
    if not valid:
        return None
    t = min(valid)
    return float(np.linalg.norm(p+v*t+.5*g*t*t-c))


def reward(state, done, cfg, robot_collision=False, arm_action=None, center=None,
           gravity=None, hit=False):
    value, terms = launch_reward(state, done, cfg, robot_collision, arm_action)
    terms.update(aim=0., basket_hit=0.)
    if state.get('confirmed') and not state.get('aim_paid'):
        candidate = state['candidate']
        error = crossing_error(candidate['position'], candidate['velocity'], center, gravity, cfg.basket_tilt_deg)
        state['aim_error'] = error
        state['aim_paid'] = True
        if error is not None:
            terms['aim'] = cfg.basket_aim_bonus * float(np.exp(-(error/cfg.basket_aim_sigma)**2))
    if hit and not state.get('hit_paid'):
        terms['basket_hit'] = cfg.basket_aim_score_bonus
        state['hit_paid'] = True
    return float(sum(terms.values())), terms
