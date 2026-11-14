"""
Continual-learning baselines on the same frozen features (gradient-trained MLP head, analytic learning, replay with
2,000 stored features, joint training). Hyperparameters are read from results/selection (select_baseline_hp.py,
selected on the default order and red-tilapia features) and reused for every order and base species.
    python scripts/run_baselines.py --feat s3d_base0 --orders default reversed random
    python scripts/run_baselines.py --feat s3d_base6 --orders base6 --methods dsal der --hp_feat s3d_base0
Output: results/baselines/<feat>/<order>/<method>_seed<s>.json and summary.json (mean and sample std)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aquagauss.common import ORDERS
from scripts.select_baseline_hp import selection_file
from vcil import methods as M
from vcil.cil import run
from vcil.features import load_split
from vcil.paths import RESULTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feat', default='s3d_base0')
    p.add_argument('--hp_feat', default='s3d_base0', help='feature set whose selected hyperparameters are used')
    p.add_argument('--orders', nargs='*', default=['default'])
    p.add_argument('--methods', nargs='*', default=list(M.GRIDS))
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44])
    a = p.parse_args()
    tr, te = load_split(a.feat, 'clip', 'train'), load_split(a.feat, 'clip', 'test')
    for oname in a.orders:
        out = RESULTS / 'baselines' / a.feat / oname
        out.mkdir(parents=True, exist_ok=True)
        summ = {}
        for m in a.methods:
            hp = json.load(open(selection_file(a.hp_feat, 'clip', m)))['best_hp']
            for s in a.seeds:
                res, _ = run(m, hp, s, tr, te, ORDERS[oname])
                for n, r in res.items():
                    r.update({'method': n, 'hp': hp, 'seed': s, 'feat': a.feat, 'order': ORDERS[oname]})
                    json.dump(r, open(out / f'{n}_seed{s}.json', 'w'), indent=1)
                    summ.setdefault(n, []).append([r['final_acc'], r['avg_inc_acc'], r['final_avg_forget']])
        table = {}
        for n, v in summ.items():
            v = 100 * np.array(v)
            table[n] = {'final': [v[:, 0].mean(), v[:, 0].std(ddof=1)], 'avg_inc': [v[:, 1].mean(), v[:, 1].std(ddof=1)],
                        'forgetting': [v[:, 2].mean(), v[:, 2].std(ddof=1)]}
            print(f'{a.feat} {oname:8s} {n:10s} final {v[:, 0].mean():.2f} ± {v[:, 0].std(ddof=1):.2f}  '
                  f'avg_inc {v[:, 1].mean():.2f}  forgetting {v[:, 2].mean():.2f}', flush=True)
        json.dump(table, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
