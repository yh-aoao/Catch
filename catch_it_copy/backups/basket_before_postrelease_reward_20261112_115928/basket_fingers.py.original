"""Phase-aware finger regularization, additive to the existing Basket objective.

No control projection or forced motion: all penalties use measured joints,
except action smoothness, which uses the policy's normalized hand commands.
"""
import numpy as np


HAND_INDICES = np.array([0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15])
DEFAULTS = dict(sync_weight=2., coupling_weight=1., extension_weight=2.,
                post_release_weight=1., smooth_weight=.01,
                sync_tolerance=.20, coupling_tolerance=.25,
                extension_margin=.05, limit_margin=.03, extension_scale=.25,
                post_release_ramp=.15, long_open=[0., 0., 0.], thumb_open=[0., 0., 0.])


def make_options(options=None):
    options = dict(options or {})
    if set(options)-set(DEFAULTS):
        raise ValueError('Unknown Basket finger options: '+str(sorted(set(options)-set(DEFAULTS))))
    result = dict(DEFAULTS, **options)
    for key in ('long_open', 'thumb_open'):
        value = np.asarray(result[key], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(key+' must contain three finite angles')
        result[key] = value.tolist()
    for key in set(DEFAULTS)-{'long_open', 'thumb_open'}:
        result[key] = float(result[key])
        if not np.isfinite(result[key]) or result[key] < 0:
            raise ValueError(key+' must be finite and nonnegative')
        if key in ('extension_scale', 'post_release_ramp') and result[key] == 0:
            raise ValueError(key+' must be positive')
    return result


def new_state(ready_joints, limits, options):
    ready = np.asarray(ready_joints, dtype=float)
    limits = np.asarray(limits, dtype=float)
    opened = np.r_[np.tile(options['long_open'], 3), options['thumb_open']]
    if ready.shape != (12,) or limits.shape != (12, 2) or not np.all(np.isfinite(ready)):
        raise ValueError('Basket finger references require twelve measured joints and limits')
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Invalid Basket finger joint limits')
    if np.any(opened < limits[:, 0]) or np.any(opened > limits[:, 1]):
        raise ValueError('Basket natural open pose exceeds joint limits')
    if np.any(ready[:9]-opened[:9] < .05):
        raise ValueError('Basket long fingers must start in a supported flexed cup')
    return dict(ready=ready.copy(), opened=opened, limits=limits.copy(),
                scale=np.maximum(np.abs(ready-opened), .10),
                pending=dict(finger_sync=0., finger_coupling=0., finger_extension=0., finger_posture=0.),
                release_time=None, previous_action=np.zeros(12), metrics={})


def update(state, measured_joints, elapsed, dt, confirmed_release, options):
    q = np.asarray(measured_joints, dtype=float)
    if (q.shape != (12,) or not np.all(np.isfinite(q)) or not np.isfinite(dt) or dt < 0
            or not np.isfinite(elapsed) or elapsed < 0):
        raise ValueError('Invalid Basket finger measurement')
    phase = ((q-state['opened'])/state['scale'])[:9].reshape(3, 3)
    # Thumb is deliberately excluded from three-long-finger synchronization.
    pairs = np.array([phase[0]-phase[1], phase[0]-phase[2], phase[1]-phase[2]])
    def bounded_square(error):
        squared = np.square(error)
        # Remain sensitive even to badly split fingers, rather than clipping
        # every large error to the same constant penalty.
        return float(np.mean(squared/(1.+squared)))
    sync = bounded_square(np.maximum(np.abs(pairs)-options['sync_tolerance'], 0.))
    coupling = bounded_square(np.maximum(np.abs(phase-phase.mean(axis=1, keepdims=True))
                                        -options['coupling_tolerance'], 0.))
    lower = state['limits'][:, 0]+options['limit_margin']
    upper = state['limits'][:, 1]-options['limit_margin']
    # Task-specific overextension applies to long fingers only. The thumb may
    # legitimately use its separate negative joint range while supporting a ball.
    lower = lower.copy()
    lower[:9] = np.maximum(lower[:9], state['opened'][:9]-options['extension_margin'])
    violation = np.maximum(lower-q, 0.)+np.maximum(q-upper, 0.)
    extension = bounded_square(violation/options['extension_scale'])
    if confirmed_release and state['release_time'] is None:
        state['release_time'] = float(elapsed)
    ramp = (0. if state['release_time'] is None else
            float(np.clip((elapsed-state['release_time'])/options['post_release_ramp'], 0., 1.)))
    error = (q-state['opened'])/state['scale']
    posture = ramp*(.75*bounded_square(error[:9])+.25*bounded_square(error[9:]))
    for key, loss, weight in (
            ('finger_sync', sync, 'sync_weight'),
            ('finger_coupling', coupling, 'coupling_weight'),
            ('finger_extension', extension, 'extension_weight'),
            ('finger_posture', posture, 'post_release_weight')):
        state['pending'][key] -= options[weight]*loss*dt
    state['metrics'] = dict(phase=phase.tolist(), sync_loss=sync, coupling_loss=coupling,
                            extension_loss=extension, post_release_ramp=ramp,
                            min_long_angle=float(np.min(q[:9])), thumb_angles=q[9:].tolist())


def reward(state, normalized_action, options):
    terms = dict(state['pending'])
    for key in state['pending']:
        state['pending'][key] = 0.
    action = np.asarray(normalized_action, dtype=float)
    if action.shape != (12,) or not np.all(np.isfinite(action)):
        raise ValueError('Basket finger smoothness requires twelve normalized actions')
    action = np.clip(action, -1., 1.)
    previous = state.get('previous_action')
    terms['finger_smoothness'] = (0. if previous is None else
        -options['smooth_weight']*float(np.mean((action-previous)**2)))
    state['previous_action'] = action.copy()
    return terms
