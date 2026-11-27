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
