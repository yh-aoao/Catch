"""Strict training contracts with explicit, recorded source-change acceptance."""
import copy


def validate_resume(saved, config, metadata, schema, allow_source_changes=False):
    adjustable = ('iterations', 'device', 'eval_every', 'eval_episodes',
                  'baseline_episodes', 'skip_teacher_baseline')
    expected, actual = copy.deepcopy(saved['config']), copy.deepcopy(config)
    for item in (expected, actual):
        for key in adjustable:
            item.pop(key, None)
    if saved['schema'] != schema or expected != actual:
        raise ValueError('Resume training/model contracts differ; start a new run')
    if saved['metadata']['teacher_hashes'] != metadata['teacher_hashes']:
        raise ValueError('Resume teacher hashes differ')
    old = saved['metadata']['source_asset_sha256']
    new = metadata['source_asset_sha256']
    changes = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
    if changes and not allow_source_changes:
        raise ValueError('Resume source/assets changed: ' + ', '.join(changes) +
                         '. Review before using --allow-source-changes.')
    reset_best = bool(changes) or any(saved['config'].get(k) != config.get(k)
                                     for k in ('eval_episodes', 'skip_teacher_baseline'))
    return changes, reset_best
