"""Joint Basket loss weighting and per-actor trust-region utilities."""
import math
import torch

DEFAULTS = dict(base_kl_limit=.02, throw_kl_limit=.01,
                replay_loss_weight=.5, reference_weight=.05)


def options(values=None):
    values=dict(values or {})
    if set(values)-set(DEFAULTS):
        raise ValueError('Unknown basket_joint_options')
    result=dict(DEFAULTS,**values)
    for key,value in result.items():
        if not math.isfinite(float(value)) or float(value)<0:
            raise ValueError(key+' must be finite and nonnegative')
        result[key]=float(value)
    if not 0<result['replay_loss_weight']<1:
        raise ValueError('replay_loss_weight must be between 0 and 1')
    if result['base_kl_limit']<=0 or result['throw_kl_limit']<=0:
        raise ValueError('Actor KL limits must be positive')
    return result


def gaussian_kl(old_mu,old_sigma,mu,sigma):
    """KL(old || new), summed over action dimensions, one value per sample."""
    return (torch.log(sigma/old_sigma)
        +(old_sigma.square()+(old_mu-mu).square())/(2.*sigma.square())-.5).sum(-1).clamp_min(0.)


def masks(raw,rehearsal):
    throwing=raw[:,47]>.5
    return ~throwing,throwing & ~rehearsal.bool(),throwing & rehearsal.bool()


def group_mean(value,mask):
    # Keep a zero attached to the graph when a group is absent.
    return (value*mask.to(value.dtype)).sum()/mask.sum().clamp_min(1)


def balanced_loss(value,groups,replay_weight,population_counts=None,population_size=None):
    navigation,mobile_throw,replay_throw=groups
    if population_counts is not None:
        # Preserve the requested loss mass over the entire rollout, even when
        # replay transitions occur in only a few minibatches.
        base_count,mobile_count,replay_count=population_counts
        if mobile_count and replay_count:
            coefficients=(1.,1.-replay_weight,replay_weight)
        else:
            coefficients=(1.,1. if mobile_count else 0.,1. if replay_count else 0.)
        total=value.sum()*0.
        for mask,count,coefficient in zip(groups,population_counts,coefficients):
            total=total+coefficient*(value*mask.to(value.dtype)).sum()/max(count,1)
        return total*population_size/max(len(value),1)
    base=group_mean(value,navigation)
    if mobile_throw.any() and replay_throw.any():
        throw=(1.-replay_weight)*group_mean(value,mobile_throw)+replay_weight*group_mean(value,replay_throw)
    else:
        throw=group_mean(value,mobile_throw | replay_throw)
    return base+throw


def normalize_advantages(raw_advantage,groups):
    result=raw_advantage.clone()
    for mask in groups:
        selected=raw_advantage[mask]
        if selected.numel()>1:
            result[mask]=(selected-selected.mean())/selected.std(unbiased=False).clamp_min(1e-8)
        elif selected.numel():
            result[mask]=selected/selected.abs().clamp_min(1.)
    return result


def bounded_lr(current,initial,kl,limit):
    """Optional KL scheduling can reduce/recover LR, never exceed its initial value."""
    if kl>limit:current/=1.5
    elif kl<limit*.5:current*=1.5
    return min(initial,max(initial*.01,current))
