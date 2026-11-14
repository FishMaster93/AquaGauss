"""
Analyses of the paper on the red-tilapia features (default order; final model = all seven species):
  significance  AquaGauss vs FeCAM, NCM and each ablation: paired bootstrap 95% CI of the accuracy difference
                (10,000 resamples of test clips) and exact McNemar test
  sensitivity   one-at-a-time sweeps of lambda, alpha, eps, beta (others fixed), validation and test
  spread        calibrated log-determinant of every cell and predicted vs true level shares
  granularity   one Gaussian per level (4 cells, species merged; still exemplar-free) vs per species-level (28)
Output: results/analysis.json
    python scripts/analyze.py --feat s3d_base0
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aquagauss.common import fit, method_configs
from aquagauss.gauss import GaussMemory
from vcil.data import PHASES
from vcil.features import load_split
from vcil.paths import RESULTS


def merge_levels(mem):
    """Merge the cells of all species into one cell per level (exact pooled mean and covariance)."""
    out = []
    for c in range(4):
        cs = [q for q in mem.cells if q['c'] == c]; N = sum(q['n'] for q in cs)
        mu = sum(q['n'] * q['mu'] for q in cs) / N
        S = sum((q['n'] - 1) * q['S'] + q['n'] * torch.outer(q['mu'] - mu, q['mu'] - mu) for q in cs) / (N - 1)
        out.append({'mu': mu, 'S': S, 'n': N, 'c': c, 'sp': -1})
    m = GaussMemory(**{k: getattr(mem, k) for k in ('mode', 'lam', 'alpha', 'eps', 'beta')})
    m.cells = out
    m.pool_num = sum((q['n'] - 1) * q['S'] for q in out); m.pool_den = sum(q['n'] - 1 for q in out)
    m.calibrate()
    return m


def main():
    p = argparse.ArgumentParser(); p.add_argument('--feat', default='s3d_base0'); a = p.parse_args()
    tr = load_split(a.feat, 'clip', 'train')
    ev = {s: load_split(a.feat, 'clip', s) for s in ('val', 'test')}
    X = {s: torch.cat([ev[s][k][0] for k in PHASES]) for s in ev}
    Y = {s: torch.cat([ev[s][k][1] for k in PHASES]) for s in ev}
    cfgs, res = method_configs(), {}
    acc = lambda mem, s: float((mem.predict(X[s]) == Y[s]).float().mean())

    # ---- significance
    pred = {m: fit(cfgs[m], tr, PHASES).predict(X['test']) for m in cfgs}
    co = (pred['aquagauss'] == Y['test']).cpu().numpy()
    B = np.random.default_rng(0).integers(0, len(co), (10000, len(co)))
    res['significance'] = {}
    for m in cfgs:
        if m == 'aquagauss' or cfgs[m] == cfgs['aquagauss']:
            continue
        cm = (pred[m] == Y['test']).cpu().numpy(); d = co.astype(float) - cm
        b, c = int((co & ~cm).sum()), int((~co & cm).sum()); bs = d[B].mean(1)
        res['significance'][m] = {'acc_aquagauss': float(co.mean()), 'acc_other': float(cm.mean()), 'diff': float(d.mean()),
                                  'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                  'mcnemar_p': float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0}
        r = res['significance'][m]
        print(f'AquaGauss vs {m:18s} {100 * r["diff"]:+.2f} [{100 * r["ci95"][0]:+.2f}, {100 * r["ci95"][1]:+.2f}] '
              f'p={r["mcnemar_p"]:.2g}', flush=True)

    # ---- sensitivity (one at a time)
    base = cfgs['aquagauss']; res['sensitivity'] = {}
    for name, vals in [('lam', [1.0, .75, .5, .33, .25]), ('alpha', [.25, .5, .6, .75, .9, 1.0]),
                       ('eps', [.003, .01, .03, .1, .3, 1.0]), ('beta', [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0, 1.25, 1.5])]:
        res['sensitivity'][name] = []
        for v in vals:
            mem = fit(dict(base, **{name: v}), tr, PHASES)
            res['sensitivity'][name].append({'value': v, 'val': acc(mem, 'val'), 'test': acc(mem, 'test')})
        print(name, [(r['value'], round(100 * r['test'], 2)) for r in res['sensitivity'][name]], flush=True)

    # ---- spread of the cells and level shares
    mem = fit(base, tr, PHASES)
    res['logdet'] = [{'species': c['sp'], 'level': c['c'], 'n': c['n'], 'logdet': float(l)} for c, l in zip(mem.cells, mem.logdet)]
    true = torch.bincount(Y['test'], minlength=4).float() / len(Y['test'])
    res['level_share'] = {'true': true.tolist()}
    for m in ('abl_t1.0_p0_maha', 'abl_t1.0_p0_ours', 'aquagauss'):
        res['level_share'][m] = (torch.bincount(pred[m], minlength=4).float() / len(pred[m])).tolist()

    # ---- granularity: 4 cells vs 28 cells (hyperparameters re-selected on validation)
    res['granularity'] = {}
    for gran in (4, 28):
        best = None
        for lam in (1.0, 0.5):
            m0 = fit({'mode': 'aquagauss', 'lam': lam}, tr, PHASES)
            m0 = merge_levels(m0) if gran == 4 else m0
            for al, e, b in itertools.product((1.0, .25, .5, .75, .9), (.01, .03, .1, .3, 1.0), (.3, .4, .5, .6, .7)):
                m0.alpha, m0.eps, m0.beta = al, e, b; m0.calibrate()
                v = acc(m0, 'val')
                if best is None or v > best[0]:
                    best = (v, acc(m0, 'test'), {'lam': lam, 'alpha': al, 'eps': e, 'beta': b})
        res['granularity'][gran] = {'val': best[0], 'test': best[1], 'cfg': best[2]}
        print(f'{gran} cells: val {100 * best[0]:.2f} test {100 * best[1]:.2f} {best[2]}', flush=True)
    json.dump(res, open(RESULTS / 'analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
