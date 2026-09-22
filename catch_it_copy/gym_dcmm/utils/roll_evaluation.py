"""Read-only Roll evaluation: never changes rewards, controls or termination."""
import numpy as np

SPEED_LIMIT = 0.15
HOLD_SECONDS = 0.30
GAP_SECONDS = 0.04
REGION_RADIUS = 0.16  # ball center to palm capture point, metres


def advance(state, contact, clear, near, speed, dt):
    valid = bool(contact and clear)
    state['intercepted'] = state.get('intercepted', False) or valid
    good = valid and near and speed <= SPEED_LIMIT
    if good:
        state['gap'] = 0.
        state['duration'] = state.get('duration', 0.) + dt
    elif clear and near and speed <= SPEED_LIMIT and not contact:
        state['gap'] = state.get('gap', 0.) + dt
        if state['gap'] > GAP_SECONDS + 1e-9:
            state['duration'] = 0.
    else:
        state['duration'], state['gap'] = 0., 0.
    held = bool(good and state.get('duration', 0.) >= HOLD_SECONDS - 1e-9)
    state['caught_once'] = state.get('caught_once', False) or held
    state['held_now'] = held
    state.update(contact=bool(contact), clear=bool(clear), near=bool(near), speed=float(speed))
    return state


def observe(env, cfg):
    if env.object_motion != 'roll':
        return
    import mujoco
    from gym_dcmm.utils.roll_rewards import hand_collision_ids
    data, model = env.Dcmm.data, env.Dcmm.model
    palm = data.body('link6')
    position = data.qpos[37:40]
    radius = float(model.geom_size[env.object_id][0])
    point = data.geom_xpos[env.hand_start_id] + palm.xmat.reshape(3, 3)[:, 1] * radius
    spatial = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, palm.id, spatial, 0)
    # MuJoCo body velocity is referenced at the body's centre of mass.
    relative = data.qvel[36:39] - spatial[3:] - np.cross(spatial[:3], position-palm.xipos)
    contacts = env.contacts['object_contacts']
    hand = bool(np.any(np.isin(contacts, hand_collision_ids(model, env.hand_start_id))))
    edge = cfg.roll_table_pos[1] - cfg.roll_table_size[1]
    clear = bool(env.table_geom_id not in contacts and env.floor_id not in contacts
                 and position[1] + radius < edge)
    state = getattr(env, '_roll_eval', {})
    env._roll_eval = advance(state, hand, clear, np.linalg.norm(position-point) <= REGION_RADIUS,
                            np.linalg.norm(relative), model.opt.timestep)


def publish(env, info, terminated, truncated):
    if env.object_motion != 'roll':
        return
    state = getattr(env, '_roll_eval', {})
    legacy = bool(info.get('success', False))
    if env.task == 'Tracking':
        # Only repair the historical flag, after reward has already been computed.
        info['success'] = bool(env.step_touch and not terminated)
        legacy = info['success']
    info['roll_legacy_success'] = legacy
    info['roll_eval_success'] = bool(state.get('intercepted' if env.task == 'Tracking' else 'caught_once', False))
    info['roll_final_hold'] = bool(state.get('held_now', False))
    info['roll_eval'] = dict(state)
    unmet = []
    for key, label in [('clear', 'ball_not_clear_of_table_or_floor'), ('contact', 'no_hand_contact')]:
        if not state.get(key, False): unmet.append(label)
    if env.task == 'Catching':
        if not state.get('near', False): unmet.append('outside_hand_region')
        if state.get('speed', float('inf')) > SPEED_LIMIT: unmet.append('relative_speed')
        if state.get('duration', 0.) < HOLD_SECONDS - 1e-9: unmet.append('hold_duration')
    info['roll_eval_unmet'] = ','.join(unmet)
    if terminated or truncated:
        info['terminated_reason'] = env.terminated_reason or ('collision_or_failure' if terminated else 'timeout')
        if env.task == 'Tracking' and env.step_touch and not terminated:
            info['terminated_reason'] = 'track_touch'
        if getattr(env, 'roll_log', False):
            print('[roll-eval] task={} legacy={} success={} final_hold={} end={} checks={} unmet={}'.format(
                env.task, legacy, info['roll_eval_success'], info['roll_final_hold'],
                info['terminated_reason'], state, unmet), flush=True)


def report(agent, infos, dones, testing):
    """Aggregate terminal metrics separately from reward-based checkpoint selection."""
    key = '_roll_eval_test_totals' if testing else '_roll_eval_train_totals'
    totals = getattr(agent, key, dict(episodes=0, success=0, final_hold=0, legacy=0))
    for i in np.flatnonzero(dones):
        final = infos.get('final_info')
        mask = infos.get('_final_info')
        if final is not None and (mask is None or mask[i]) and isinstance(final[i], dict):
            record = final[i]
        else:
            record = {k: np.asarray(infos[k])[i] for k in ('roll_eval_success', 'roll_final_hold', 'roll_legacy_success') if k in infos}
        if 'roll_eval_success' not in record: continue
        totals['episodes'] += 1
        totals['success'] += int(record['roll_eval_success'])
        totals['final_hold'] += int(record.get('roll_final_hold', False))
        totals['legacy'] += int(record.get('roll_legacy_success', False))
        if totals['episodes'] == 1 or totals['episodes'] % 20 == 0:
            n = totals['episodes']
            print('[roll-summary] mode={} episodes={} eval_success={:.3f} final_hold={:.3f} legacy_success={:.3f}'.format(
                'test' if testing else 'train', n, totals['success']/n, totals['final_hold']/n, totals['legacy']/n), flush=True)
    setattr(agent, key, totals)
