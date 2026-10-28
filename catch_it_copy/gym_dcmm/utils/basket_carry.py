"""Additive carry shaping; original launch, aim and basket rewards stay intact."""
import numpy as np


DEFAULTS = dict(forward_distance=.06, forward_speed=.30, forward_bonus=6.,
                position_bonus=2., slip_cost=2., release_bonus=4.,
                local_tolerance=[.008, .015, .010], local_sigma=[.015, .020, .015])


def make_options(options=None):
    options = dict(options or {})
    if set(options)-set(DEFAULTS):
        raise ValueError('Unknown Basket carry reward options')
    result = dict(DEFAULTS, **options)
    for key in ('local_tolerance', 'local_sigma'):
        value = np.asarray(result[key], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(key+' must contain three positive finite distances')
        result[key] = value.tolist()
    for key in set(DEFAULTS)-{'local_tolerance', 'local_sigma'}:
        result[key] = float(result[key])
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(key+' must be positive and finite')
    return result


def new_state(palm_relative, ball_local):
    return dict(origin=np.asarray(palm_relative, dtype=float).copy(),
                local_reference=np.asarray(ball_local, dtype=float).copy(), frontier=0.,
                quality_integral=0., forward_pending=0., position_pending=0., slip_pending=0.)


def update(state, palm_relative, proximal_velocity, ball_velocity, ball_local, direction, options):
    """Call only while supported and before the first confirmed launch.

    Reward each new forward distance at most once. Stable holding without
    transport, arm motion after release, base-only motion and wrist-only ball
    acceleration cannot accumulate positive carry credit.
    """
    forward = float(np.dot(np.asarray(palm_relative)-state['origin'], direction))
    arm_speed = float(np.dot(proximal_velocity, direction))
    ball_speed = float(np.dot(ball_velocity, direction))
    excess = np.maximum(np.abs(np.asarray(ball_local)-state['local_reference'])
                        - np.asarray(options['local_tolerance']), 0.)
    position_quality = float(np.exp(-np.sum((excess/options['local_sigma'])**2)))
    progress = float(np.clip(forward/options['forward_distance'], 0., 1.))
    delta = max(0., progress-state['frontier'])
    state['frontier'] = max(state['frontier'], progress)
    # Smooth nonzero guidance for the small movements of the pretrained policy.
    moving = float(np.clip(arm_speed/options['forward_speed'], 0., 1.)
                   * np.clip(ball_speed/options['forward_speed'], 0., 1.))
    state['forward_pending'] += options['forward_bonus']*delta*moving
    state['position_pending'] += options['position_bonus']*delta*moving*position_quality
    state['slip_pending'] -= options['slip_cost']*delta*(1.-position_quality)
    state['quality_integral'] += delta*position_quality
    state.update(forward=forward, proximal_speed=arm_speed, ball_speed=ball_speed,
                 position_quality=position_quality, local_error=excess.tolist())


def measure(model, data, state, center, options):
    import mujoco
    from gym_dcmm.utils.basket_grip import palm_coordinates
    palm, base = data.body('link6'), data.body('arm_base')
    direction = np.asarray(center)-base.xpos
    direction[2] = 0.
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jac, None, palm.id)
    columns = [int(model.jnt_dofadr[np.flatnonzero(model.jnt_qposadr == q)[0]]) for q in (15,16,17)]
    update(state, palm.xpos-base.xpos, jac[:,columns] @ data.qvel[columns],
           data.qvel[36:39], palm_coordinates(data, data.qpos[37:40]), direction, options)


def reward(state, launch, options, collision=False):
    terms = dict(carry_forward=state.pop('forward_pending', 0.),
                 carry_position=state.pop('position_pending', 0.),
                 carry_slip=state.pop('slip_pending', 0.), carry_release=0.)
    # Pending fields must remain usable by the next physics step.
    state.update(forward_pending=0., position_pending=0., slip_pending=0.)
    if collision:
        return dict.fromkeys(terms, 0.)
    if launch.get('confirmed') and not state.get('release_paid'):
        distance = float(np.clip(state.get('forward',0.)/options['forward_distance'], 0., 1.))
        speed = float(np.clip(state.get('proximal_speed',0.)/options['forward_speed'], 0., 1.))
        support = state['quality_integral']/max(state['frontier'], 1e-9)
        terms['carry_release'] = options['release_bonus']*np.sqrt(distance)*speed*np.sqrt(support)
        state['release_paid'] = True
    return {key:float(value) for key,value in terms.items()}
