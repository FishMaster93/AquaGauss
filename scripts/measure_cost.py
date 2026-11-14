"""
Deployment cost of AquaGauss: memory of the stored statistics, time to add a species (statistics + recalibration,
GPU), and per-clip inference time (batch size 1, median after warm-up) of the encoder and of the scoring step,
on the GPU and on the CPU with four threads.
    python scripts/measure_cost.py --feat s3d_base0 --ckpt workspace/S3D_base0/phase0_best.pt
Output: results/cost.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aquagauss.common import ORDERS, method_configs, run_incremental
from vcil.features import load_split
from vcil.models.video import VideoModel
from vcil.paths import RESULTS


def bench(f, sync, warm, n):
    for _ in range(warm):
        f()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); f(); sync(); ts.append(1000 * (time.perf_counter() - t0))
    return statistics.median(ts)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feat', default='s3d_base0'); p.add_argument('--ckpt', default=None); p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    tr, te = load_split(a.feat, 'clip', 'train'), load_split(a.feat, 'clip', 'test')
    r = run_incremental(method_configs()['aquagauss'], tr, te, ORDERS['default'])
    res = {'memory_MB': r['memory_MB'], 'add_species_s': float(np.mean(r['update_sec'][1:]))}
    from aquagauss.common import fit
    mem = fit(method_configs()['aquagauss'], tr, ORDERS['default'])
    x1 = te[0][0][:1].contiguous()
    res['score_gpu_ms'] = bench(lambda: mem.predict(x1), torch.cuda.synchronize, 50, 1000)
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        model = VideoModel(head=ck['args'].get('head', 'coral'), backbone=ck.get('backbone', 's3d')); model.load_state_dict(ck['model_state_dict'])
        enc = model.backbone.eval(); clip = torch.randint(0, 256, (1, 16, 224, 224, 3), dtype=torch.uint8)
        torch.backends.cudnn.benchmark = True
        enc_g, clip_g = enc.cuda(), clip.cuda()
        res['encoder_gpu_ms'] = bench(lambda: enc_g(clip_g), torch.cuda.synchronize, 20, 100)
        torch.set_num_threads(a.threads)
        enc = enc.cpu()
        res[f'encoder_cpu{a.threads}_ms'] = bench(lambda: enc(clip), lambda: None, 3, 20)
    json.dump(res, open(RESULTS / 'cost.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
