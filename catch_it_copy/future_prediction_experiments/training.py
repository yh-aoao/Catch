"""Schedule and non-invasive replay-batch diagnostics, no new policy inputs."""
import math
import torch
from torch.nn import functional as F


def learning_rate(config,iteration):
    sched = config['lr_schedule']
    if sched['kind'] == 'constant':
        return config['learning_rate']
    start,end = sched['start'],sched['end']
    progress = min(1.,max(0.,(iteration-start)/(end-start)))
    return sched['minimum'] + .5*(config['learning_rate']-sched['minimum'])*(1+math.cos(math.pi*progress))


@torch.no_grad()
def diagnostics(student,batch,begin,end,logits,config,acc):
    valid = batch['valid'][:,begin:end]
    target = batch['target'][:,begin:end]
    if config['loss'] == 'bce':
        error = 2*F.binary_cross_entropy_with_logits(2*logits,(target+1)/2,reduction='none')
    else:
        error = (logits.tanh()-target).square()
    groups = [(0,2),(2,8)]+([(8,20)] if target.shape[-1] == 20 else [])
    imitation = torch.stack([error[...,a:b].mean(-1) for a,b in groups]).mean(0)
    mask = batch['future_valid'][:,begin:end]*valid[...,None]
    mu = student.extra['future'].reshape(*mask.shape,3)
    squared = (mu-batch['future_target'][:,begin:end]).square().mean(-1)
    horizons = config['model'].get('future_prediction',{}).get('horizons',[1,3,5])
    def record(prefix,frames):
        def add(key,value):
            acc[key] = acc.get(key,0.)+value.item()
        add(prefix+'/frames',frames.sum())
        add(prefix+'/imitation_sum',(imitation*frames).sum())
        for k,horizon in enumerate(horizons):
            m = mask[...,k]*frames
            label = prefix+'/h'+str(horizon)
            add(label+'/count',m.sum())
            add(label+'/squared_m_sum',(squared[...,k]*m).sum())
            if 'sigma' in student.extra:
                sigma = student.extra['sigma'].reshape(*mask.shape,3)[...,k,:]
                add(label+'/sigma_m_sum',(sigma.mean(-1)*m).sum())
                covered = ((mu[...,k,:]-batch['future_target'][:,begin:end,k,:]).abs() <= 1.96*sigma).float().mean(-1)
                add(label+'/coverage95_sum',(covered*m).sum())
    per_task = config['batch_episodes_per_task']
    # BaseReplay.sample yields contiguous groups in configured task order.
    for index,task in enumerate(config['tasks']):
        frames = torch.zeros_like(valid)
        frames[index*per_task:(index+1)*per_task] = valid[index*per_task:(index+1)*per_task]
        record('diagnostic/'+task,frames)
        if task == 'bounce':
            for phase,name in enumerate(('pre_rebound_proxy','post_rebound_proxy','near_hand','post_hand_contact')):
                record('diagnostic/bounce/'+name,frames*(batch['forecast_phase'][:,begin:end]==phase))


def finish_diagnostics(acc):
    out = dict(acc)
    for key,value in acc.items():
        prefix = key.rsplit('/',1)[0]
        if key.endswith('/imitation_sum'):
            out[prefix+'/imitation_mean'] = value/max(1.,acc[prefix+'/frames'])
        if key.endswith('/squared_m_sum'):
            out[prefix+'/rmse_m'] = math.sqrt(value/max(1.,acc[prefix+'/count']))
        for name in ('sigma_m','coverage95'):
            if key.endswith('/'+name+'_sum'):
                out[prefix+'/'+name+'_mean'] = value/max(1.,acc[prefix+'/count'])
    return out
