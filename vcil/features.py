"""
Frozen-encoder feature cache: features/<name>/phase{p}.pt = {'paths': [...], 'feat': (N, d) fp16, 'meta': {...}}.
One feature set per encoder checkpoint; train/val/test are selected at load time from the manifest.
"""
import torch

from vcil.data import PHASES, phase_items
from vcil.paths import FEAT_ROOT


def feat_file(name, phase):
    return FEAT_ROOT / name / f'phase{phase}.pt'


def load_split(name, protocol, split, phases=None, device='cuda'):
    """{phase: (x float32 (N, d), y long (N,))}, in manifest order."""
    out = {}
    for ph in phases if phases is not None else PHASES:
        d = torch.load(feat_file(name, ph), weights_only=False)
        idx = {p: i for i, p in enumerate(d['paths'])}
        items = phase_items(ph, split, protocol)
        missing = [it['video'] for it in items if it['video'] not in idx]
        assert not missing, f'{name} phase{ph}: {len(missing)} clips without features, e.g. {missing[0]}'
        sel = torch.tensor([idx[it['video']] for it in items])
        out[ph] = (d['feat'][sel].float().to(device), torch.tensor([it['label'] for it in items], device=device))
    return out
