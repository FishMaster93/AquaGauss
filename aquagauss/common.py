"""Helpers shared by the scripts: species orders, fitting a memory, evaluation."""
import json
import time

import numpy as np
import torch

from aquagauss.gauss import GaussMemory
from vcil.paths import RESULTS

# Species are numbered as in Table 1: 0 red tilapia, 1 tilapia, 2 jade perch, 3 largemouth bass, 4 lotus carp,
# 5 sunfish, 6 O. punctatus. The first species of an order is the base species on which the encoder was adapted.
ORDERS = {'default': [0, 1, 2, 3, 4, 5, 6], 'reversed': [0, 6, 5, 4, 3, 2, 1], 'random': [0, 3, 6, 1, 5, 2, 4],
          'base6': [6, 0, 1, 2, 3, 4, 5], 'base5': [5, 0, 1, 2, 3, 4, 6]}
HPARAMS = RESULTS / 'hparams.json'


def method_configs():
    """AquaGauss (selected on validation by scripts/select_hparams.py), its ablations, FeCAM and NCM."""
    cfgs = json.load(open(HPARAMS))['configs']
    cfgs['fecam'] = {'mode': 'fecam', 'lam': 0.5, 'g1': 1.0}
    cfgs['ncm'] = {'mode': 'ncm', 'lam': 0.5}
    return cfgs


def fit(cfg, train, species):
    mem = GaussMemory(**cfg)
    for sp in species:
        mem.add_species(*train[sp], species=sp)
    return mem


def run_incremental(cfg, train, test, order):
    """Add species one at a time; after each phase evaluate on the test clips of all learned species."""
    mem, acc, upd = GaussMemory(**cfg), np.full((7, 7), np.nan), []
    for t, sp in enumerate(order):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        mem.add_species(*train[sp], species=sp)
        torch.cuda.synchronize(); upd.append(time.perf_counter() - t0)
        for j, sj in enumerate(order[:t + 1]):
            x, y = test[sj]
            acc[t, j] = float((mem.predict(x) == y).float().mean())
    m = np.array([len(test[s][1]) for s in order])
    A = [float((acc[t, :t + 1] * m[:t + 1]).sum() / m[:t + 1].sum()) for t in range(7)]
    forg = float(np.mean([acc[j:, j].max() - acc[6, j] for j in range(6)]))   # Eq. 12 (max includes last phase)
    return {'cfg': cfg, 'order': order, 'acc': acc.tolist(), 'A': A, 'final': A[-1], 'avg_inc': float(np.mean(A)),
            'forgetting': forg, 'update_sec': upd, 'memory_MB': mem.memory_bytes() / 2 ** 20}
