"""
Data definition for video-only species-incremental FFIR. All splits come from data/manifest_video.csv
(built by scripts/build_manifest.py). protocol='clip' is the clip-level 80/10/10 split (column split_clip).
Each sample is keyed by its .npy path; every clip is a preprocessed (16, 224, 224, 3) uint8 array.
"""
import csv
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from vcil.paths import MANIFEST

PHASES = {
    0: ['Red_tilapia_2', 'Red_tilapia_3'],
    1: ['Tilapia'],
    2: ['Jade_perch'],
    3: ['Largemouth_bass'],
    4: ['Lotus_carp'],
    5: ['Sunfish'],
    6: ['Oplegnathus_punctatus'],
}
CLASS_NAMES = ['None', 'Weak', 'Medium', 'Strong']
C = len(CLASS_NAMES)


def split_col(protocol):
    if protocol == 'clip':
        return 'split_clip'
    if protocol.startswith('session'):
        return f'split_s{protocol[len("session"):] or "0"}'
    raise ValueError(f'unknown protocol: {protocol}')


@lru_cache(maxsize=1)
def load_manifest():
    with open(MANIFEST, newline='') as f:
        return list(csv.DictReader(f))


def phase_items(phase, split=None, protocol='clip'):
    """Samples of one phase; split=None returns all clips of the phase (for feature extraction)."""
    species = set(PHASES[phase])
    col = split_col(protocol) if split else None
    return [{'video': r['video_path'], 'label': int(r['label']), 'species': r['species'], 'session': r['session']}
            for r in load_manifest()
            if r['species'] in species and (split is None or r[col] == split)]


class VideoClipDataset(Dataset):
    """Reads preprocessed (16, 224, 224, 3) uint8 clips; augmentation is done on the GPU."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        clip = torch.from_numpy(np.load(it['video']))
        return clip, it['label'], i


def gpu_augment(x, *paired, scale=(0.8, 1.0)):
    """x: (B,T,H,W,3) uint8 on GPU -> same shape. Per sample: random horizontal flip + random crop resized back.
    Tensors in `paired` get exactly the same augmentation; returns (x', *paired')."""
    B, T, H, W, _ = x.shape
    outs = [torch.empty_like(t) for t in (x, *paired)]
    for i in range(B):
        s = np.random.uniform(*scale)
        h, w = int(H * s), int(W * s)
        top, left = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
        flip = np.random.rand() < 0.5
        for src, out in zip((x, *paired), outs):
            clip = src[i][:, top:top + h, left:left + w, :]
            if flip:
                clip = clip.flip(2)
            clip = clip.permute(0, 3, 1, 2).float()
            clip = F.interpolate(clip, size=(H, W), mode='bilinear', align_corners=False)
            out[i] = clip.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return outs[0] if not paired else tuple(outs)
