"""
Analytic (closed-form) continual learning on frozen video features, exemplar-free: a fixed random projection
Linear(d, D) + ReLU (Kaiming normal, fixed seed, never trained) followed by ridge regression; each phase adds the
current species to A = sum x x^T and B = sum x y^T and solves again.

  acil : one-hot ridge regression [Zhuang et al., NeurIPS 2022]
  dsal : DS-AL [Zhuang et al., AAAI 2024]: main stream (ReLU) + compensation stream (Tanh, fits the residual of the
         main stream); prediction = main + C * compensation
"""
import torch
import torch.nn.functional as F

from vcil.data import C

DEV = 'cuda'


class RandomBuffer:
    """Fixed random projection; weights initialised like nn.Linear (out, in) (fan_in = in), stored as (in, out)."""

    def __init__(self, in_dim, out_dim, seed, device=DEV):
        g = torch.Generator().manual_seed(seed)
        w = torch.empty(out_dim, in_dim)
        torch.nn.init.kaiming_normal_(w, generator=g)
        self.w = w.T.contiguous().to(device)

    def __call__(self, x):
        return torch.relu(x @ self.w)


def aug(x):
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)


class Ridge:
    def __init__(self, dim, lam, n_out):
        self.A = torch.zeros(dim, dim, device=DEV)
        self.B = torch.zeros(dim, n_out, device=DEV)
        self.lam, self.W = lam, None

    def solve(self):
        self.W = torch.linalg.solve(self.A + self.lam * torch.eye(len(self.A), device=DEV), self.B)

    def __call__(self, X):
        return X @ self.W


class AnalyticLearner:
    def __init__(self, method, d_in, hp, seed, D=8192):
        self.method, self.hp = method, hp
        self.buf = RandomBuffer(d_in, D, seed)
        self.tanh_buf = RandomBuffer(d_in, D, seed + 1)
        self.main = Ridge(D + 1, hp['lam'], C)
        if method == 'dsal':
            self.comp = Ridge(D + 1, hp['lam'], C)

    def phi(self, x, kind='relu'):
        return aug(self.buf(x) if kind == 'relu' else torch.tanh(x @ self.tanh_buf.w))

    def learn(self, x, y, chunk=8192):
        for i in range(0, len(y), chunk):
            X = self.phi(x[i:i + chunk])
            self.main.A += X.T @ X
            self.main.B += X.T @ F.one_hot(y[i:i + chunk], C).float()
        self.main.solve()
        if self.method == 'dsal':   # compensation stream fits the residual of the main stream on the current species
            for i in range(0, len(y), chunk):
                xx, yy = x[i:i + chunk], y[i:i + chunk]
                res = F.one_hot(yy, C).float() - self.main(self.phi(xx))
                Xc = self.phi(xx, 'tanh')
                self.comp.A += Xc.T @ Xc
                self.comp.B += Xc.T @ res
            self.comp.solve()

    @torch.no_grad()
    def predict(self, x, variant=None, chunk=4096):
        out = []
        for i in range(0, len(x), chunk):
            xx = x[i:i + chunk]
            s = self.main(self.phi(xx))
            if self.method == 'dsal':
                s = s + self.hp['C'] * self.comp(self.phi(xx, 'tanh'))
            out.append(s.argmax(1))
        return torch.cat(out)
