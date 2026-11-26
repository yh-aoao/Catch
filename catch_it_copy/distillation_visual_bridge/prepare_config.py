"""Explicitly pin the four selected teacher files before a fresh experiment.
Never silently accept changed teacher content during training or resume.
"""
import argparse,copy,hashlib,json,os
from pathlib import Path
from .core import TASKS


def pin(config,teacher_root=None,bounce_checkpoint=None,basket_checkpoint=None):
    result=copy.deepcopy(config);resolved={}
    for task in TASKS:
        entry=result['tasks'][task];path=Path(entry['checkpoint']).expanduser()
        if teacher_root is not None:path=Path(teacher_root).expanduser()/path.name
        if task=='bounce' and bounce_checkpoint is not None:path=Path(bounce_checkpoint).expanduser()
        if task=='basket' and basket_checkpoint is not None:path=Path(basket_checkpoint).expanduser()
        if not path.is_absolute():path=Path(__file__).resolve().parents[1]/path
        path=path.resolve()
        if not path.is_file():raise FileNotFoundError('{} teacher missing: {}'.format(task,path))
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        entry.update(checkpoint=str(path),sha256=digest.hexdigest());resolved[task]=path
    # This experiment reuses the controlled-hold Bounce environment, so reject a known incompatible teacher.
    import torch
    saved=torch.load(resolved['bounce'],map_location='cpu',weights_only=False)
    expected=result['tasks']['bounce']['env_kwargs'].get('bounce_grasp_version','legacy')
    actual=saved.get('bounce_grasp_version','legacy')
    if actual!=expected:raise ValueError('Bounce grasp protocol mismatch: file={} config={}'.format(actual,expected))
    saved=torch.load(resolved['basket'],map_location='cpu',weights_only=False)
    if saved.get('schema')!='basket_mobile_bridge_v1':raise ValueError('Expected Bridge Basket teacher')
    result['tasks']['basket']['env_kwargs']['options']=dict(saved['config']['options'],evaluation_task='far_mobile',basket_diameter_m=.6)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/catch.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--teacher-root',type=Path);p.add_argument('--bounce-checkpoint',type=Path);p.add_argument('--basket-checkpoint',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new prepared-config filename')
    config=pin(json.loads(a.config.read_text()),a.teacher_root,a.bounce_checkpoint,a.basket_checkpoint)
    from .run import validate
    validate(config)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_name(a.output.name+'.tmp');temporary.write_text(json.dumps(config,indent=2)+'\n');os.replace(temporary,a.output)
    for task in TASKS:print(task,config['tasks'][task]['checkpoint'],config['tasks'][task]['sha256'])
    print('Prepared config:',a.output.resolve())

if __name__=='__main__':main()
