"""Test-only episode diagnostics, including vector autoreset terminal info."""
import json
from collections import Counter
import numpy as np

def report(agent, infos, dones):
    totals = getattr(agent, '_failure_diagnostic_totals', {})
    for i in np.flatnonzero(dones):
        final = infos.get('final_info')
        mask = infos.get('_final_info')
        if final is not None and (mask is None or mask[i]) and isinstance(final[i], dict):
            raw = final[i].get('failure_diagnostic')
        else:
            values = infos.get('failure_diagnostic')
            present = infos.get('_failure_diagnostic')
            raw = values[i] if values is not None and (present is None or present[i]) else None
        if not raw:
            continue
        record = json.loads(raw)
        task = record['task']
        stats = totals.setdefault(task, dict(episodes=0, failures=0, reasons=Counter(), unmet=Counter()))
        stats['episodes'] += 1
        if not record['success']:
            stats['failures'] += 1
            stats['reasons'][record['end']] += 1
            stats['unmet'].update(set(record['unmet']))
            print('[test-failure] ' + raw, flush=True)
        print('[test-failure-summary] task={} episodes={} failures={} success_rate={:.3f} end_counts={} unmet_counts={}'.format(
            task, stats['episodes'], stats['failures'], 1-stats['failures']/stats['episodes'], dict(stats['reasons']), dict(stats['unmet'])), flush=True)
    agent._failure_diagnostic_totals = totals
