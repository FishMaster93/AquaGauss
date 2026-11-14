"""
Registry of the continual-learning baselines (learner, hyperparameter grid). All run on frozen features.
NCM, FeCAM and AquaGauss itself are implemented in aquagauss/gauss.py.
"""
from vcil.methods.analytic import AnalyticLearner
from vcil.methods.grad import GradLearner

EXEMPLAR_FREE = ['finetune', 'lwf', 'ewc', 'si', 'mas', 'acil', 'dsal']
EXEMPLAR = ['er', 'icarl', 'der', 'agem', 'gdumb']     # store old samples (here: frozen features)
UPPER = ['joint']

B = 2000    # memory budget (number of clips) of the exemplar-based methods
GRIDS = {
    'finetune': [{'epochs': e, 'lr': lr} for e in (10, 30) for lr in (1e-3, 3e-4)],
    'lwf': [{'epochs': 10, 'lr': 1e-3, 'kd': kd} for kd in (0.5, 1, 2, 5)],
    'ewc': [{'epochs': 10, 'lr': 1e-3, 'reg': w} for w in (1e2, 1e3, 1e4, 1e5)],
    'si': [{'epochs': 10, 'lr': 1e-3, 'reg': w} for w in (1, 10, 100, 1e3, 1e4)],
    'mas': [{'epochs': 10, 'lr': 1e-3, 'reg': w} for w in (1, 10, 100, 1e3)],
    # the grid extends down to 1e-2 so that the selected value is not on the edge
    'acil': [{'lam': l} for l in (1e-2, 0.1, 1, 10, 100, 1e3, 1e4)],
    'dsal': [{'lam': l, 'C': c} for l in (1e-2, 0.1, 1, 10, 100, 1e3) for c in (0.5, 1.0)],
    'er': [{'epochs': e, 'lr': 1e-3, 'budget': B} for e in (10, 30)],
    'icarl': [{'epochs': 10, 'lr': 1e-3, 'kd': kd, 'budget': B} for kd in (0.5, 1, 2)],
    'der': [{'epochs': 10, 'lr': 1e-3, 'alpha': a, 'beta': b, 'budget': B} for a in (0.01, 0.1, 0.5) for b in (0.5, 1.0, 2.0)],
    'agem': [{'epochs': e, 'lr': 1e-3, 'budget': B} for e in (10, 30)],
    'gdumb': [{'epochs': e, 'lr': 1e-3, 'budget': B} for e in (30, 100, 300)],
    'joint': [{'epochs': e, 'lr': lr} for e in (10, 30) for lr in (1e-3, 3e-4)],
}
assert set(GRIDS) == set(EXEMPLAR_FREE + EXEMPLAR + UPPER)

# methods that report several results from one run (iCaRL: NME and FC classifiers of the same model)
VARIANTS = {'icarl': ['nme', 'fc']}

_LEARNERS = {'acil': AnalyticLearner, 'dsal': AnalyticLearner}


def build(method, d_in, hp, seed):
    return _LEARNERS.get(method, GradLearner)(method, d_in, hp, seed)
