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
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PPO_Catch_TwoStage')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ns = {'torch': torch, 'np': np}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns[name]


class Teacher:
    def __init__(self, config, device='cpu'):
        self.device = torch.device(device)
        path = ROOT / config['checkpoint']
        self.hash = sha256(path)
        if config.get('sha256') and self.hash != config['sha256']:
            raise ValueError('Teacher checkpoint hash mismatch: ' + str(path))
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['model']
        self.action_dim = state['mu_t.weight'].shape[0] + state['mu_c.weight'].shape[0]
        if state['mu_c.weight'].shape[0] != 12:
            raise ValueError('Only full TwoStage Catch teachers with twelve hand outputs are supported')
        family = ROOT / 'gym_dcmm' / 'algs' / config['family']
        self.family = family
        units = [v.shape[0] for k, v in state.items()
                 if k.startswith('actor_mlp_t.mlp.') and k.endswith('.weight')]
        self.model = load_file(family / 'models_catch.py').ActorCritic({
            'actor_units': units, 'actions_num': self.action_dim,
            'tracking_actions_num': self.action_dim - 12, 'input_shape': (30,),
            'separate_value_mlp': True}).to(self.device)
        self.model.load_state_dict(state, strict=True)
        rms = load_file(family / 'utils.py').RunningMeanStd
        self.track, self.hand = rms((18,)).to(self.device), rms((12,)).to(self.device)
        self.track.load_state_dict(checkpoint['running_mean_std_track'], strict=True)
        self.hand.load_state_dict(checkpoint['running_mean_std_hand'], strict=True)
        for module in (self.model, self.track, self.hand):
            module.eval().requires_grad_(False)
            if any(not torch.isfinite(t).all() for t in module.state_dict().values()):
                raise ValueError('Non-finite teacher parameters/statistics')
        if (self.track.running_var < 0).any() or (self.hand.running_var < 0).any():
            raise ValueError('Negative normalization variance')

    @torch.no_grad()
    def predict_tensor(self, raw):
        z = torch.cat([self.track(raw[:, :18]), self.hand(raw[:, 18:])], -1)
        return self.model.act_inference({'obs': z, 'obs_t': z[:, :18], 'obs_c': z[:, 2:]}).clamp(-1, 1)

    def predict(self, raw):
        return self.predict_tensor(torch.as_tensor(raw, dtype=torch.float32, device=self.device)).cpu().numpy()

    @torch.no_grad()
    def parity_check(self, raw):
        raw = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
        proxy = SimpleNamespace(model=self.model, running_mean_std_track=self.track,
                                running_mean_std_hand=self.hand)
        reference = original_method(self.family / 'ppo_dcmm_catch_two_stage.py', 'model_act')(
            proxy, {'obs': raw}, inference=True)['actions'].clamp(-1, 1)
        error = (reference - self.predict_tensor(raw)).abs().max().item()
        if error > 1e-5:
            raise ValueError('Teacher inference differs from original model_act: ' + str(error))
        return error
