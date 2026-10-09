"""Read-only terminal diagnostics; never alter success or simulator termination."""
import numpy as np


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def terminal_diagnostics(info):
    keys = ('track_touch', 'track_contact_success', 'success', 'roll_eval_success', 'roll_legacy_success', 'roll_final_hold',
            'roll_eval', 'roll_eval_unmet', 'roll_hold', 'env_time', 'ee_distance',
            'terminated_reason', '_terminated', '_truncated')
    return {k: json_value(info[k]) for k in keys if k in info}


def summarize(rows, metric, requested):
    n = sum(row['success'] for row in rows)
    total = len(rows)
    reasons = {}
    for row in rows:
        reasons[row['reason']] = reasons.get(row['reason'], 0) + 1
    auxiliary = {}
    for key in ('track_contact_success', 'success', 'roll_eval_success', 'roll_legacy_success', 'roll_final_hold'):
        values = [r['diagnostics'][key] for r in rows if key in r['diagnostics']]
        if values:
            auxiliary[key] = {'true': sum(bool(v) for v in values), 'reported': len(values),
                              'rate': sum(bool(v) for v in values) / len(values)}
    return {'success_metric': metric, 'successes': n, 'episodes': total,
            'requested_episodes': requested, 'complete': total == requested,
            'success_rate': n / total, 'auxiliary_metrics': auxiliary,
            'termination_counts': reasons,
            'legacy_truncated_rate': sum(r['legacy_truncated'] for r in rows) / total,
            'mean_reward': sum(r['reward'] for r in rows) / total, 'records': rows}
