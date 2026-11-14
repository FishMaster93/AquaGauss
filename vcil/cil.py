"""
Species-incremental loop and metrics for the baselines. After each phase the learner is evaluated on the test
clips of all species learned so far:
  acc_seen   accuracy over all seen test clips (A_t)        avg_inc_acc  mean of A_t over the 7 phases
  avg_forget mean over earlier species of (best accuracy so far - current accuracy)
"""
import numpy as np

from vcil import methods as M
from vcil.data import PHASES


def default_order(order=None):
    """Species order. A 6-element order lists phases 1-6 (species 0 first); a 7-element order is used as given
    (its first species is the base species on which the encoder was adapted)."""
    if order and len(order) == len(PHASES):
        return list(order)
    order = [0] + (list(order) if order else [p for p in PHASES if p != 0])
    assert sorted(order) == sorted(PHASES), order
    return order


def run(method, hp, seed, train, evals, order=None):
    """train / evals: {phase: (x, y)}. Returns ({result name: {summary..., history}}, learner)."""
    order = default_order(order)
    learner = M.build(method, train[order[0]][0].shape[1], hp, seed)
    variants = M.VARIANTS.get(method, [None])
    hist = {v: [] for v in variants}
    best = {v: {} for v in variants}
    for step, ph in enumerate(order):
        zs = {}
        if step > 0:
            ex, ey = evals[ph]
            zs = {v: float((learner.predict(ex, v) == ey).float().mean()) for v in variants}
        learner.learn(*train[ph])
        for v in variants:
            accs, corr, tot = {}, 0, 0
            for q in order[:step + 1]:
                ex, ey = evals[q]
                c = int((learner.predict(ex, v) == ey).sum())
                accs[q] = c / len(ey)
                corr, tot = corr + c, tot + len(ey)
            for q, a in accs.items():
                best[v][q] = max(best[v].get(q, 0.0), a)
            fg = float(np.mean([best[v][q] - accs[q] for q in order[:step]])) if step else 0.0
            hist[v].append({'step': step, 'phase': ph, 'species': PHASES[ph], 'acc_seen': corr / tot,
                            'avg_forget': fg, 'zero_shot': zs.get(v), 'per_phase': accs})
    res = {}
    for v in variants:
        h = hist[v]
        res[method if v is None else f'{method}_{v}'] = {
            'final_acc': h[-1]['acc_seen'], 'avg_inc_acc': float(np.mean([x['acc_seen'] for x in h])),
            'final_avg_forget': h[-1]['avg_forget'], 'history': h}
    return res, learner
