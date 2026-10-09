"""Strict inference-only restoration, without constructing PPO or a simulator."""
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_file(path):
    spec = importlib.util.spec_from_file_location('distill_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_method(path, name):
    """Extract an existing pure inference method for parity checks, avoiding PPO imports."""
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PPO_Track')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ns = {'torch': torch, 'np': np}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns[name]


class Teacher:
    def __init__(self, config, device='cpu'):
        self.device = torch.device(device)
        path = ROOT / config['checkpoint']
        if not path.is_file():
            raise FileNotFoundError('Teacher checkpoint not found: {}. Copy the configured teacher '
                                    'to this machine or run on the training server; no fallback to old weights.'.format(path))
        self.hash = sha256(path)
        if config.get('sha256') and self.hash != config['sha256']:
            raise ValueError('Teacher checkpoint hash mismatch: ' + str(path))
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['model']
        self.action_dim = state['mu.weight'].shape[0]
        if self.action_dim != config['expected_action_dim']:
            raise ValueError('Unexpected teacher action dimension: ' + str(self.action_dim))
        if state['actor_mlp.mlp.0.weight'].shape[1] != 18:
            raise ValueError('Expected 18D track teacher observation')
        self.family = ROOT / 'gym_dcmm' / 'algs' / config['family']
        units = [v.shape[0] for k, v in state.items()
                 if k.startswith('actor_mlp.mlp.') and k.endswith('.weight')]
        self.model = load_file(self.family / 'models_track.py').ActorCritic({
            'actor_units': units, 'actions_num': self.action_dim, 'input_shape': (18,),
            'separate_value_mlp': any(k.startswith('value_mlp.') for k in state)}).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.rms = load_file(self.family / 'utils.py').RunningMeanStd((18,)).to(self.device)
        self.rms.load_state_dict(checkpoint['running_mean_std'], strict=True)
        for module in (self.model, self.rms):
            module.eval().requires_grad_(False)
            if any(not torch.isfinite(t).all() for t in module.state_dict().values()):
                raise ValueError('Non-finite teacher parameters/statistics')
        if (self.rms.running_var < 0).any():
            raise ValueError('Negative normalization variance')

    @torch.no_grad()
    def predict_tensor(self, raw):
        return self.model.act_inference({'obs': self.rms(raw)}).clamp(-1, 1)

    def predict(self, raw):
        return self.predict_tensor(torch.as_tensor(raw, dtype=torch.float32, device=self.device)).cpu().numpy()

    @torch.no_grad()
    def parity_check(self, raw):
        raw = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        proxy = SimpleNamespace(model=self.model, running_mean_std=self.rms)
        reference = original_method(self.family / 'ppo_dcmm_track.py', 'model_act')(
            proxy, {'obs': raw}, inference=True)['actions'].clamp(-1, 1)
        error = (reference - self.predict_tensor(raw)).abs().max().item()
        if error > 1e-5:
            raise ValueError('Teacher inference differs from original model_act: ' + str(error))
        return error
