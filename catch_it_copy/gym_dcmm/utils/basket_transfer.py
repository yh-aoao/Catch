"""Explicit fixed35/arm18 -> mobile51/base2+arm18 actor migration."""
import hashlib
from pathlib import Path

import torch


def migrate_actor(model, normalizer, checkpoint):
    if (checkpoint.get('action_layout') != 'arm6_hand12' or
            checkpoint.get('basket_arm_control') != 'joint_velocity'):
        raise ValueError('Basket warm start requires a fixed-base joint_velocity arm6_hand12 checkpoint')
    source = checkpoint['model']
    target = model.state_dict()
    if (source['actor_mlp.mlp.0.weight'].shape[1] != 35 or
            source['mu.weight'].shape[0] != 18 or
            target['actor_mlp.mlp.0.weight'].shape[1] != 51 or
            target['mu.weight'].shape[0] not in (18, 20)):
        raise ValueError('Basket migration requires fixed35/18 -> mobile51/(18 or 20)')
    offset = target['mu.weight'].shape[0] - 18
    if not model.separate_value_mlp:
        raise ValueError('Basket migration requires separate actor/value MLPs; mobile critic starts fresh')
    norm = checkpoint.get('running_mean_std')
    if norm is None or norm['running_mean'].shape != (35,):
        raise ValueError('Basket migration requires original 35-dimensional observation normalization')
    # Prepare every tensor before mutating the model, and reject hidden-size changes.
    for key, value in source.items():
        if not key.startswith('actor_mlp.'):
            continue
        if key == 'actor_mlp.mlp.0.weight':
            if target[key].shape[0] != value.shape[0]:
                raise ValueError('Basket actor hidden widths must match the source')
            target[key] = torch.zeros_like(target[key])
            target[key][:, :35] = value.to(target[key])
        else:
            if key not in target or target[key].shape != value.shape:
                raise ValueError('Basket actor hidden widths must match the source: ' + key)
            target[key] = value.to(target[key]).clone()
    for key in ('mu.weight', 'mu.bias', 'sigma'):
        if target[key].shape[1:] != source[key].shape[1:]:
            raise ValueError('Basket output hidden width mismatch')
        target[key] = torch.zeros_like(target[key])
        target[key][offset:] = source[key].to(target[key])
    # Fresh base exploration; old arm/finger standard deviations are preserved.
    target['sigma'][:offset] = -.7
    stats = normalizer.state_dict()
    stats['running_mean'] = torch.zeros_like(stats['running_mean'])
    stats['running_var'] = torch.ones_like(stats['running_var'])
    stats['running_mean'][:35] = norm['running_mean'].to(stats['running_mean'])
    stats['running_var'][:35] = norm['running_var'].to(stats['running_var'])
    # Fixed task's unused parking-Y feature was +.2 at reset. The mobile feature
    # is zero at the desired stand. Shift the mean to preserve that reference.
    stats['running_mean'][34] -= .2
    # Critic, value normalization and mobile feature statistics stay fresh.
    model.load_state_dict(target, strict=True)
    normalizer.load_state_dict(stats, strict=True)
    normalizer.eval()
    return dict(version='fixed35_arm18_to_mobile51_{}_v1'.format('base2_arm18' if offset else 'arm18'),
                freeze_input_normalization=True, legacy_parking_y_offset=.2)


def from_file(model, normalizer, filename, device='cpu'):
    checkpoint = torch.load(filename, map_location=device)
    metadata = migrate_actor(model, normalizer, checkpoint)
    metadata.update(source_name=Path(filename).name,
                    source_sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest())
    return metadata
