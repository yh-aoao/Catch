"""Rendering placement is independent of learning and physical task contracts."""
from model_experiments.visual_resume import validate_resume as base_validate_resume

RUNTIME_KEYS=('render_devices','render_task_devices')

def validate_render_config(config):
    devices=config.get('render_devices')
    mapping=config.get('render_task_devices')
    if devices is not None and mapping is not None:
        raise ValueError('Use either render_devices or render_task_devices')
    if devices is not None and (not isinstance(devices,list) or not devices or
            any(type(d) is not int or d<0 for d in devices) or len(set(devices))!=len(devices)):
        raise ValueError('render_devices must be unique nonnegative EGL device IDs')
    if mapping is not None and (not isinstance(mapping,dict) or set(mapping)!=set(config['tasks']) or
            any(type(d) is not int or d<0 for d in mapping.values())):
        raise ValueError('render_task_devices must specify all four tasks with nonnegative EGL IDs')
    if (devices is not None or mapping is not None) and config['vision']['gl_backend'] not in ('egl','auto'):
        raise ValueError('Render device assignment requires EGL backend')

def render_device(config,task,index,count):
    mapping=config.get('render_task_devices')
    if mapping is not None:return mapping[task]
    devices=config.get('render_devices')
    if devices is None:return None
    offset=list(config['tasks']).index(task)*count
    return devices[(offset+index)%len(devices)]

def validate_resume(saved,config,metadata,schema,allow_source_changes=False):
    # Hardware placement may change without changing the sampling/training contract.
    saved=dict(saved,config={k:v for k,v in saved['config'].items() if k not in RUNTIME_KEYS})
    actual={k:v for k,v in config.items() if k not in RUNTIME_KEYS}
    return base_validate_resume(saved,actual,metadata,schema,allow_source_changes)


def latch_episode_choices(choices,latched=None):
    import numpy as np
    if latched is None:latched=np.full(len(choices),-1,dtype=np.int8)
    fresh=latched<0
    latched[fresh]=choices[fresh].astype(np.int8)
    return latched,latched.astype(bool)


def validate_initial_student(saved,config,schema,teacher_hashes):
    """Weight-only transfer into a separately named repair experiment."""
    import copy
    if saved.get('schema')!=schema:raise ValueError('Initialization schema mismatch')
    old=saved['config']
    if old.get('ablation','full')!=config.get('ablation','full'):raise ValueError('Initialization ablation mismatch')
    if saved['metadata']['teacher_hashes']!=teacher_hashes:raise ValueError('Initialization teacher hashes differ')
    for key in ('model','vision','env_kwargs','tasks'):
        a,b=copy.deepcopy(old[key]),copy.deepcopy(config[key])
        if key=='vision':
            a.pop('gl_backend',None);b.pop('gl_backend',None)
        if key=='tasks':
            for entries in (a,b):
                for item in entries.values():
                    item.pop('checkpoint',None);item.pop('success_metric',None)
        if a!=b:raise ValueError('Initialization task/model contract differs: '+key)


def validate_torch_device(requested):
    """Validate CUDA logical ordinals before model allocation; never remap silently."""
    import os
    import torch
    device=torch.device(requested)
    if device.type!='cuda':return device
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    count=torch.cuda.device_count()
    if not torch.cuda.is_available() or count==0:
        raise RuntimeError('CUDA unavailable in this process: visible GPU count={}, CUDA_VISIBLE_DEVICES={!r}. Check the assigned GPU session and driver.'.format(count,visible))
    index=device.index if device.index is not None else torch.cuda.current_device()
    if index>=count:
        valid=', '.join('cuda:'+str(i) for i in range(count))
        raise ValueError('Invalid network device {}: PyTorch sees {} GPU(s); valid logical devices: {}. CUDA_VISIBLE_DEVICES={!r}. CUDA ordinals are logical after visibility filtering: e.g. CUDA_VISIBLE_DEVICES=6 usually makes physical GPU6 cuda:0. EGL numbering is separate; do not infer EGL IDs from this count.'.format(device,count,valid,visible))
    return torch.device('cuda',index)
