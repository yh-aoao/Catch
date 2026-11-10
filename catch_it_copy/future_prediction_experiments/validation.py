import math


def validate_experiment(config):
    if type(config['evaluation_seed']) is not int or config['evaluation_seed'] < 0:
        raise ValueError('evaluation_seed must be a nonnegative integer')
    s = config['lr_schedule']
    if s['kind'] not in ('constant','cosine'):
        raise ValueError('Unknown LR schedule')
    if s['kind'] == 'cosine':
        if not (type(s['start']) is int and type(s['end']) is int and 0 <= s['start'] < s['end']):
            raise ValueError('Invalid schedule boundaries')
        if not math.isfinite(s['minimum']) or not 0 < s['minimum'] <= config['learning_rate']:
            raise ValueError('Invalid minimum learning rate')
    p = config['model'].get('future_prediction')
    if p:
        h = p['horizons']
        if not h or any(type(x) is not int or x<1 for x in h) or sorted(set(h)) != h:
            raise ValueError('Horizons must be sorted unique positive integers')
        if h[-1] >= config['max_episode_steps']:
            raise ValueError('Forecast horizon exceeds maximum episode length')
        if not math.isfinite(p['target_scale_m']) or p['target_scale_m'] <= 0:
            raise ValueError('target_scale_m must be positive')
        lo,hi = p['logvar_bounds']
        if not (math.isfinite(lo) and math.isfinite(hi) and -20 <= lo < 0 < hi <= 10):
            raise ValueError('Invalid bounded log variance')
    elif config['experiment']['variant'] == 'future_uncertain':
        raise ValueError('Missing uncertainty configuration')
