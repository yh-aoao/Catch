"""Training-only contact event telemetry and stratified loss sampling.

Observe existing contact queries without advancing simulation or changing rewards.
Sampling selects loss positions; the complete episode is still forwarded through GRU.
"""
import numpy as np
import torch

NAMES = ('table_separation', 'ground_contact', 'first_hand_contact', 'sustained_hand_contact')


class ContactEvents:
    def __init__(self, env):
        self.env = env
        self.original = env._get_contacts
        env._get_contacts = self.observe
        self.reset()

    def reset(self):
        self.flags = np.zeros(len(NAMES), np.float32)
        self.table_seen = self.table_previous = self.floor_previous = False
        self.hand_seen = self.hold_seen = False
        self.hand_since = None

    def begin_step(self):
        self.flags[:] = 0

    def observe(self):
        contacts = self.original()
        ids = np.asarray(contacts['object_contacts'], dtype=int)
        env = self.env
        table = getattr(env, 'table_geom_id', -1)
        floor = getattr(env, 'floor_id', -1)
        on_table = table >= 0 and bool(np.any(ids == table))
        on_floor = floor >= 0 and bool(np.any(ids == floor))
        hand = bool(np.any((ids >= env.hand_start_id) & (ids < env.object_id)))
        now = float(env.Dcmm.data.time)
        if self.table_seen and self.table_previous and not on_table:
            self.flags[0] = 1
        if on_floor and not self.floor_previous:
            self.flags[1] = 1
        if hand and not self.hand_seen:
            self.flags[2] = 1
            self.hand_seen = True
        if hand:
            if self.hand_since is None:
                self.hand_since = now
            if now - self.hand_since >= .12 and not self.hold_seen:
                self.flags[3] = 1
                self.hold_seen = True
        else:
            self.hand_since = None
        self.table_seen |= on_table
        self.table_previous, self.floor_previous = on_table, on_floor
        return contacts


def sample_positions(flags, valid, config):
    """Equal budget per episode; event half is uniform over available event types.

    Draw with replacement so short windows remain eligible. Counts are loss weights;
    padding is never selected. Zero-event episodes use uniform valid positions.
    """
    counts = torch.zeros_like(valid)
    for row in range(len(valid)):
        positions = torch.where(valid[row] > 0)[0]
        if not len(positions):
            raise ValueError('Cannot sample an empty episode')
        if not config['enabled']:
            counts[row, positions] = 1
            continue
        windows = []
        for kind in range(flags.shape[-1]):
            centers = torch.where((flags[row, :, kind] > 0) & (valid[row] > 0))[0]
            if len(centers):
                delta = positions[:, None] - centers[None, :]
                mask = ((delta >= -config['pre_steps']) & (delta <= config['post_steps'])).any(1)
                windows.append(positions[mask])
        total = config['samples_per_episode']
        special = round(total * config['event_fraction']) if windows else 0
        chosen = [positions[torch.randint(len(positions), (total-special,), device=valid.device)]]
        for _ in range(special):
            candidates = windows[int(torch.randint(len(windows), ()).item())]
            chosen.append(candidates[torch.randint(len(candidates), (1,), device=valid.device)])
        indices = torch.cat(chosen)
        counts[row].scatter_add_(0, indices, torch.ones_like(indices, dtype=valid.dtype))
    return counts
