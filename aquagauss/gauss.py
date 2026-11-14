"""
AquaGauss: exemplar-free species-incremental FFIR with a Gaussian distribution memory.

Every species-intensity cell k = (species, level) is stored as (mu_k, Sigma_k, n_k), computed once from the
power-transformed features of its training clips; no clip or feature is kept afterwards.

  power transform   z = max(u, 0) ** lam                                   (Eq. 1)
  cell statistics   mu_k, Sigma_k (unbiased), n_k                          (Eqs. 2-3)
  pooled covariance Sigma_pool = sum (n_j - 1) Sigma_j / sum (n_j - 1)     (Eq. 4, running sums)
  shrinkage         Sigma'_k  = alpha Sigma_k + (1 - alpha) Sigma_pool     (Eq. 5)
  ridge             Sigma''_k = Sigma'_k + eps * tr(Sigma'_k) / d * I      (Eq. 6)
  score             s_k(z) = (z - mu_k)^T Sigma''_k^-1 (z - mu_k) + beta * log|Sigma''_k|   (Eq. 7)
  prediction        level of argmin_k s_k(z)                               (Eq. 9)

The same class also provides the two statistics-based baselines used in the paper:
  mode='ncm'   : nearest cell mean (Euclidean) on the power-transformed features
  mode='fecam' : FeCAM (Goswami et al., 2023): L2-normalised features, Sigma + g1 * mean(diag) * I,
                 correlation normalisation, Mahalanobis score
"""
import torch
import torch.nn.functional as F

N_LEVELS = 4


def power_transform(x, lam):
    return x if lam == 1.0 else x.clamp(min=0).pow(lam)


def mahalanobis(mus, precs, x, chunk=4096):
    out = []
    for i in range(0, len(x), chunk):
        xx = x[i:i + chunk]
        out.append(torch.stack([(((xx - m) @ P) * (xx - m)).sum(1) for m, P in zip(mus, precs)], 1))
    return torch.cat(out)


class GaussMemory:
    def __init__(self, mode='aquagauss', lam=0.5, alpha=0.75, eps=0.03, beta=0.5, g1=1.0):
        assert mode in ('aquagauss', 'fecam', 'ncm')
        self.mode, self.lam, self.alpha, self.eps, self.beta, self.g1 = mode, lam, alpha, eps, beta, g1
        self.cells = []                      # list of dicts: mu, S (float64), n, c (level), sp (species)
        self.pool_num = self.pool_den = None  # running sums of (n - 1) * Sigma and (n - 1)
        self._ready = False

    # ---------------------------------------------------------------- learning
    def _features(self, x):
        z = power_transform(x, self.lam)
        return F.normalize(z, dim=1) if self.mode == 'fecam' else z

    @torch.no_grad()
    def add_species(self, x, y, species):
        """x: (N, d) frozen encoder features of the new species, y: (N,) levels 0..3."""
        z = self._features(x)
        for c in range(N_LEVELS):
            zc = z[y == c].double()
            cell = {'mu': zc.mean(0), 'S': torch.cov(zc.T), 'n': len(zc), 'c': c, 'sp': species}
            self.cells.append(cell)
            num, den = (cell['n'] - 1) * cell['S'], cell['n'] - 1
            self.pool_num = num if self.pool_num is None else self.pool_num + num
            self.pool_den = den if self.pool_den is None else self.pool_den + den
        self.calibrate()

    @torch.no_grad()
    def calibrate(self):
        """Recompute the scoring matrices of all cells (Eqs. 4-6); stored statistics are never changed."""
        self.mus = [c['mu'].float() for c in self.cells]
        self.levels = torch.tensor([c['c'] for c in self.cells], device=self.cells[0]['mu'].device)
        if self.mode == 'ncm':
            self.M = torch.stack(self.mus)
        elif self.mode == 'fecam':
            self.precs = []
            for c in self.cells:
                S = c['S']; I = torch.eye(len(S), device=S.device, dtype=S.dtype)
                Ss = S + self.g1 * S.diagonal().mean() * I
                sd = Ss.diagonal().sqrt()
                self.precs.append(torch.linalg.inv(Ss / torch.outer(sd, sd)).float())
        else:
            pool = self.pool_num / self.pool_den
            self.precs, logdet = [], []
            for c in self.cells:
                S = self.alpha * c['S'] + (1 - self.alpha) * pool
                S = S + self.eps * S.diagonal().mean() * torch.eye(len(S), device=S.device, dtype=S.dtype)
                self.precs.append(torch.linalg.inv(S).float())
                logdet.append(torch.linalg.slogdet(S)[1].item())
            self.logdet = torch.tensor(logdet, device=self.levels.device)
        self._ready = True

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def scores(self, x):
        """(N, K) scores against all learned cells, lower is better."""
        z = self._features(x)
        if self.mode == 'ncm':
            return torch.cdist(z, self.M) ** 2
        s = mahalanobis(self.mus, self.precs, z)
        return s + self.beta * self.logdet if self.mode == 'aquagauss' else s

    @torch.no_grad()
    def predict(self, x):
        return self.levels[self.scores(x).argmin(1)]

    # ---------------------------------------------------------------- bookkeeping
    def memory_bytes(self, dtype_bytes=4):
        """Stored statistics in fp32: per cell the mean and the upper triangle of the covariance,
        plus the pooled scatter for AquaGauss (NCM: means only)."""
        d = len(self.cells[0]['mu']); tri = d * (d + 1) // 2
        per_cell = d if self.mode == 'ncm' else d + tri
        return (len(self.cells) * per_cell + (tri if self.mode == 'aquagauss' else 0)) * dtype_bytes
