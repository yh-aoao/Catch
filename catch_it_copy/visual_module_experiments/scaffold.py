"""Reproducible generation of isolated runners/configs from the auxiliary baseline.

Run once when implementing, not as part of training. Existing runs are never copied.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'future': 'distillation_aux_future',
    'belief': 'distillation_aux_belief',
    'spatial': 'distillation_aux_spatial',
    'contact': 'distillation_aux_contact',
    'camera_gate': 'distillation_aux_camera_gate',
}


def main():
    for variant,name in VARIANTS.items():
        dest = ROOT/name
        (dest/'configs').mkdir(parents=True,exist_ok=True)
        (dest/'__init__.py').write_text('',encoding='utf8')
        (dest/'core.py').write_text(
            'from distillation_visual_aux.core import *\n'
            'from visual_module_experiments.core import Student as ModuleStudent, EpisodeReplay\n'
            'SCHEMA = '+repr(name+'_v1')+'\n'
            'class Student(ModuleStudent):\n    variant = '+repr(variant)+'\n',encoding='utf8')
        run = (ROOT/'distillation_visual_aux/run.py').read_text(encoding='utf8')
        run = run.replace('distillation_visual_aux',name)
        run = run.replace("('"+name+"', 'distillation',", "('"+name+"', 'visual_module_experiments', 'distillation_visual_aux', 'model_experiments', 'distillation',")
        run = run.replace("cfg.update(stage=config['stage'], vision=config['vision'])",
                          "cfg.update(stage=config['stage'], vision=config['vision'])\n    cfg['extra_labels'] = config.get('experiment', {}).get('variant') in ('belief', 'spatial')")
        run = run.replace("'object_valid': packet['object_valid'][i].copy()}",
            "'object_valid': packet['object_valid'][i].copy(),\n"
            "                                    'world_position': packet['world_position'][i].copy(),\n"
            "                                    'uv': packet['uv'][i].copy(), 'visible': packet['visible'][i].copy(),\n"
            "                                    'contact_after': np.float32(0)}")
        run = run.replace("_, _, done, _ = pool.step(adapter.action_dict(command))",
            "_, _, done, infos = pool.step(adapter.action_dict(command))\n"
            "                for i, pending in enumerate(self.pending[task]):\n"
            "                    pending[-1]['contact_after'] = np.float32(infos[i]['module_hand_contact'])", 1)
        run = run.replace('velocity_loss=0.)','velocity_loss=0., module_loss=0.)')
        run = run.replace("        if not torch.isfinite(loss):", "        module_loss = student.extra_loss(batch, begin, end, lengths)\n        loss = loss + module_loss\n        if not torch.isfinite(loss):")
        run = run.replace("('velocity_loss', aux['velocity'])]", "('velocity_loss', aux['velocity']), ('module_loss', module_loss)]")
        # Eval has no privileged module labels or extra segmentation renders.
        run = run.replace("        pool = Pool(task, env_config(config, task), 1,", "        evaluation_env = env_config(config, task)\n        evaluation_env['extra_labels'] = False\n        pool = Pool(task, evaluation_env, 1,")
        run = run.replace("    adapters = original.make_adapters(config)",
            "    if config.get('experiment', {}).get('variant') != "+repr(variant)+":\n"
            "        raise ValueError('Config belongs to a different module experiment')\n"
            "    adapters = original.make_adapters(config)")
        (dest/'run.py').write_text(run,encoding='utf8')
        env = (ROOT/'distillation_visual_aux/environments.py').read_text(encoding='utf8')
        env = env.replace('from .core import encode_depth, camera_specs',
            'from .core import encode_depth, camera_specs\nfrom visual_module_experiments.labels import labels, hand_contact')
        env = env.replace('        episode = 0',
            "        contact_seen = False\n"
            "        original_contacts = env._get_contacts\n"
            "        def recorded_contacts():\n"
            "            nonlocal contact_seen\n"
            "            value = original_contacts()\n"
            "            contact_seen = contact_seen or hand_contact(env)\n"
            "            return value\n"
            "        env._get_contacts = recorded_contacts\n"
            "        episode = 0")
        env = env.replace("            previous_object, previous_time = position, before",
            "            result.update(labels(env, sensor, specs, config.get('extra_labels', False)))\n"
            "            previous_object, previous_time = position, before")
        env = env.replace('            obs, reward, terminated, truncated, info = env.step(payload)',
            '            contact_seen = False\n            obs, reward, terminated, truncated, info = env.step(payload)')
        env = env.replace("            info = dict(info)", "            info = dict(info)\n            info['module_hand_contact'] = contact_seen or hand_contact(env)")
        (dest/'environments.py').write_text(env,encoding='utf8')
        for stage in ('track','catch'):
            cfg=json.loads((ROOT/'distillation_visual_aux/configs'/ (stage+'.json')).read_text(encoding='utf8'))
            cfg.update(iterations=10000,eval_every=2000,eval_episodes=300,baseline_episodes=300,
                       num_envs_per_task=4,experiment={'variant':variant},skip_teacher_baseline=True)
            cfg['vision']['width']=cfg['vision']['height']=160
            cfg['auxiliary'].update(position_weight=.1,velocity_weight=0.)
            (dest/'configs'/(stage+'.json')).write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
        (dest/'README.md').write_text('# '+name+'\n\n'
            'Independent C-series variant: `'+variant+'`, on the B6 position-auxiliary baseline.\n'
            'See [shared design, tests and commands](../visual_module_experiments/README.md).\n'
            'Run: `python -m '+name+'.run train --stage catch --device cuda:0 --gl-backend egl '
            '--iterations 10000 --eval-every 2000 --episodes 300 --num-envs 4 '
            '--skip-teacher-baseline --output '+name+'/runs/catch_10000_seed123`\n',encoding='utf8')


if __name__ == '__main__':
    main()
