"""Do not optimize action probabilities for masked, unexecuted navigation actions."""
import torch


def masked_mean(value, mask):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32, device=mask.device)
    if value.ndim == 0:
        return value
    value = value.reshape(len(mask), -1).mean(-1)
    weight = mask.to(value.dtype)
    return (value*weight).sum()/weight.sum().clamp_min(1.)


def clear_actor_gradients(model):
    # None (rather than a zero tensor) also prevents Adam momentum updates.
    for module in (model.actor_mlp, model.mu):
        for parameter in module.parameters():
            parameter.grad = None
    model.sigma.grad = None
