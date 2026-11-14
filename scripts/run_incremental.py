"""
Species-incremental runs of AquaGauss, its ablations, FeCAM and NCM on frozen features.
Hyperparameters come from results/hparams.json and are not re-tuned for other orders or base species.
    python scripts/run_incremental.py --feat s3d_base0 --orders default reversed random
    python scripts/run_incremental.py --feat s3d_base6 --orders base6
Output: results/incremental/<feat>/<order>.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aquagauss.common import ORDERS, method_configs, run_incremental
from vcil.features import load_split
from vcil.paths import RESULTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feat', default='s3d_base0')
    p.add_argument('--orders', nargs='*', default=['default', 'reversed', 'random'])
    p.add_argument('--methods', nargs='*', default=None, help='default: all (aquagauss, fecam, ncm, ablations)')
    a = p.parse_args()
    tr, te = load_split(a.feat, 'clip', 'train'), load_split(a.feat, 'clip', 'test')
    cfgs = method_configs()
    out = RESULTS / 'incremental' / a.feat
    out.mkdir(parents=True, exist_ok=True)
    for oname in a.orders:
        res = {'order': ORDERS[oname], 'feat': a.feat}
        for m in a.methods or cfgs:
            r = res[m] = run_incremental(cfgs[m], tr, te, ORDERS[oname])
            print(f'{a.feat} {oname:8s} {m:18s} final {100 * r["final"]:.2f} avg_inc {100 * r["avg_inc"]:.2f} '
                  f'forgetting {100 * r["forgetting"]:.2f}  update {np.mean(r["update_sec"][1:]):.2f}s '
                  f'memory {r["memory_MB"]:.2f} MB', flush=True)
        json.dump(res, open(out / f'{oname}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
