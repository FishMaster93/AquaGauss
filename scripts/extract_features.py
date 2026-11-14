"""
Extract frozen features of every clip with the phase-0 encoder (global average pooling of the last block,
d = 1024 for S3D). bf16 autocast, stored in fp16; no augmentation.
    python scripts/extract_features.py --name s3d_base0 --ckpt workspace/S3D_base0/phase0_best.pt
Output: features/<name>/phase{p}.pt = {'paths': [...], 'feat': (N, d) fp16, 'meta': {...}}
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil.data import PHASES, VideoClipDataset, phase_items
from vcil.features import feat_file
from vcil.models.video import VideoModel


@torch.no_grad()
def extract(encoder, items, batch_size, num_workers):
    loader = DataLoader(VideoClipDataset(items), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    feats = []
    for x, _, _ in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            feats.append(encoder(x.cuda(non_blocking=True))['embedding'].half().cpu())
    return torch.cat(feats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--ckpt', required=True, help='workspace/<exp>/phase0_best.pt from train_phase0.py')
    p.add_argument('--batch_size', type=int, default=48)
    p.add_argument('--num_workers', type=int, default=10)
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model = VideoModel(head=ck['args'].get('head', 'coral'), backbone=ck.get('backbone', 's3d'))
    model.load_state_dict(ck['model_state_dict'])
    enc = model.backbone.cuda().eval()
    for ph in PHASES:
        out = feat_file(a.name, ph)
        if out.exists():
            print('skip', out); continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0, items = time.time(), phase_items(ph)
        f = extract(enc, items, a.batch_size, a.num_workers)
        torch.save({'paths': [it['video'] for it in items], 'feat': f,
                    'meta': {'name': a.name, 'phase': ph, 'ckpt': a.ckpt, 'backbone': ck.get('backbone', 's3d')}}, out)
        print(f'phase {ph}: {len(items)} clips {tuple(f.shape[1:])} {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
