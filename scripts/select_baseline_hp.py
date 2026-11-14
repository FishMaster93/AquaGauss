"""
Hyperparameter selection for the baselines: every configuration of vcil/methods GRIDS is run on the validation set
(seed 42) and the one with the best average incremental accuracy is kept (iCaRL: FC classifier).
Output: results/selection/<feat>_<protocol>/<method>.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcil import methods as M
from vcil.cil import run
from vcil.features import load_split
from vcil.paths import RESULTS


def selection_file(feat, protocol, method):
    return RESULTS / 'selection' / f'{feat}_{protocol}' / f'{method}.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feat', default='s3d_base0')
    p.add_argument('--protocol', default='clip')
    p.add_argument('--methods', nargs='*', default=list(M.GRIDS))
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    train, val = load_split(args.feat, args.protocol, 'train'), load_split(args.feat, args.protocol, 'val')
    for method in args.methods:
        out = selection_file(args.feat, args.protocol, method)
        if out.exists():
            print(f'skip {out}')
            continue
        grid = []
        for hp in M.GRIDS[method]:
            t0 = time.time()
            res, _ = run(method, hp, args.seed, train, val)
            r = res.get(f'{method}_fc', res.get(method))
            grid.append({'hp': hp, 'S': r['avg_inc_acc'], 'final_acc': r['final_acc'],
                         'final_avg_forget': r['final_avg_forget']})
            print(f'  {method:10s} {hp} val avg_inc {100 * r["avg_inc_acc"]:.2f} final {100 * r["final_acc"]:.2f} '
                  f'| {time.time() - t0:.0f}s', flush=True)
        best = max(grid, key=lambda g: g['S'])
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'method': method, 'feat': args.feat, 'protocol': args.protocol, 'criterion': 'val avg_inc_acc',
                       'best_hp': best['hp'], 'grid': grid}, f, indent=2)
        print(f'{method}: best {best["hp"]} (val avg_inc {100 * best["S"]:.2f})', flush=True)


if __name__ == '__main__':
    main()
