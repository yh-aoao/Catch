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


def restore_cuda_rng(states, saved_device=None, device=None):
    """Restore visible generators, then remap the training stream across devices."""
    import torch
    if states is None or not torch.cuda.is_available():
        return
    count = torch.cuda.device_count()
    for index, value in enumerate(states[:count]):
        torch.cuda.set_rng_state(value.cpu(), index)
    if saved_device is not None and device is not None:
        source, target = torch.device(saved_device), torch.device(device)
        if source.type == 'cuda' and target.type == 'cuda':
            src = source.index if source.index is not None else 0
            dst = target.index if target.index is not None else torch.cuda.current_device()
            if dst >= count:
                raise ValueError('Requested CUDA device is not visible: ' + str(target))
            if src >= len(states):
                raise ValueError('Checkpoint lacks RNG state for its training device: ' + str(source))
            torch.cuda.set_rng_state(states[src].cpu(), dst)
            if src != dst or len(states) != count:
                print('[resume-rng] restored training RNG {} -> {}; saved GPUs={}, visible GPUs={}'.format(
                    source, target, len(states), count), flush=True)
