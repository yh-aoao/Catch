"""Frozen actor hidden features; hooks exist only while collecting teacher labels."""
import numpy as np
import torch
from torch.nn import functional as F


def predict_with_features(teacher, raw, width):
    model = teacher.model
    names = ['actor_mlp'] if hasattr(model, 'actor_mlp') else ['actor_mlp_t', 'actor_mlp_c']
    captured, handles = {}, []
    try:
        for name in names:
            def hook(module, inputs, output, key=name):
                captured[key] = F.normalize(output.detach(), dim=-1)
            handles.append(getattr(model, name).register_forward_hook(hook))
        with torch.no_grad():
            action = teacher.predict(raw)
        feature = F.normalize(torch.cat([captured[name] for name in names], -1), dim=-1)
        if feature.shape[-1] > width:
            raise ValueError('Teacher feature exceeds configured feature_width; increase it before training')
        if not torch.isfinite(feature).all():
            raise ValueError('Nonfinite teacher feature')
        target = np.zeros((len(raw), width), np.float32)
        target[:, :feature.shape[-1]] = feature.cpu().numpy()
        return action, target
    finally:
        for handle in handles:
            handle.remove()


def feature_loss(student, target, task_indices, valid, lengths):
    # Task ID selects a training-only alignment head, never the deployed policy.
    per_step = torch.zeros_like(valid)
    for index, task in enumerate(student.feature_tasks):
        prediction = F.normalize(student.feature_heads[task](student.recurrent_features), dim=-1)
        error = (prediction - target).square().sum(-1)
        per_step = per_step + error * (task_indices == index).to(valid.dtype)
    return ((per_step * valid).sum(1) / lengths.clamp_min(1)).mean()
