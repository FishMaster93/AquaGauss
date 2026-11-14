"""
Select the hyperparameters of AquaGauss and of its ablations on the validation set (final accuracy).

  power lambda in {1 (none), 0.5}; pooled shrinkage: no (alpha = 1) or alpha in {0.25, 0.5, 0.75, 0.9};
  ridge eps in {0.01, 0.03, 0.1, 0.3, 1}; score: Mahalanobis (beta = 0), spread-aware (beta in {0.3..0.7}),
  or full Gaussian likelihood (beta = 1).

Each of the 2 x 2 x 3 combinations keeps its own best setting. 'aquagauss' = power 0.5 + shrinkage + spread-aware.
The test set is only evaluated for reporting. Output: results/hparams.json
    python scripts/select_hparams.py --feat s3d_base0
"""
import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aquagauss.common import HPARAMS
from aquagauss.gauss import mahalanobis, power_transform
from aquagauss.common import fit
from vcil.data import PHASES
from vcil.features import load_split

ALPHA, EPS, BETA = (0.25, 0.5, 0.75, 0.9), (0.01, 0.03, 0.1, 0.3, 1.0), (0.3, 0.4, 0.5, 0.6, 0.7)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--feat', default='s3d_base0'); a = p.parse_args()
    tr = load_split(a.feat, 'clip', 'train')
    ev = {s: load_split(a.feat, 'clip', s) for s in ('val', 'test')}
    rows, configs = [], {}
    for lam in (1.0, 0.5):
        mem = fit({'mode': 'aquagauss', 'lam': lam}, tr, PHASES)
        z = {s: power_transform(torch.cat([ev[s][k][0] for k in PHASES]), lam) for s in ev}
        y = {s: torch.cat([ev[s][k][1] for k in PHASES]) for s in ev}
        cand = {}
        for al, e in itertools.product((1.0,) + ALPHA, EPS):
            mem.alpha, mem.eps = al, e
            mem.calibrate()
            D = {s: mahalanobis(mem.mus, mem.precs, z[s]) for s in ev}
            for score, betas in (('maha', (0.0,)), ('ours', BETA), ('qda1', (1.0,))):
                for b in betas:
                    acc = {s: float((mem.levels[(D[s] + b * mem.logdet).argmin(1)] == y[s]).float().mean()) for s in ev}
                    cand.setdefault((al < 1.0, score), []).append((acc['val'], acc['test'], (al, e, b)))
        for (pool, score), lst in cand.items():
            v, t, (al, e, b) = max(lst, key=lambda r: r[0])
            name = f'abl_t{lam}_p{int(pool)}_{score}'
            configs[name] = {'mode': 'aquagauss', 'lam': lam, 'alpha': al, 'eps': e, 'beta': b}
            rows.append({'name': name, 'val': v, 'test': t, **configs[name]})
            print(f'{name:18s} val {100 * v:.2f} test {100 * t:.2f}  alpha={al} eps={e} beta={b}', flush=True)
    configs['aquagauss'] = configs['abl_t0.5_p1_ours']
    HPARAMS.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'feat': a.feat, 'criterion': 'validation final accuracy', 'configs': configs, 'rows': rows},
              open(HPARAMS, 'w'), indent=1)
    print('AquaGauss:', configs['aquagauss'])


if __name__ == '__main__':
    main()
