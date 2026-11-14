"""
Gradient-based continual learning on frozen video features; all methods share the same classifier head:
LayerNorm -> Linear(d, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 4)

Exemplar-free:
  finetune : sequential fine-tuning
  lwf      : Learning without Forgetting [Li & Hoiem, TPAMI 2017], distillation of the old outputs (T = 2)
  ewc      : online EWC [Kirkpatrick et al., PNAS 2017], diagonal empirical Fisher
  si       : Synaptic Intelligence [Zenke et al., ICML 2017], path-integral importance
  mas      : Memory Aware Synapses [Aljundi et al., ECCV 2018], importance from gradients of the output L2 norm
With exemplars (fixed total budget split equally over species x level cells; frozen features are stored):
  er       : experience replay, random selection, current data concatenated with the memory
  icarl    : herding selection + distillation [Rebuffi et al., CVPR 2017]; NME and FC classifiers reported
  der      : DER++ [Buzzega et al., NeurIPS 2020]: memory also stores logits, loss += a*MSE(logits) + b*CE(memory)
  agem     : A-GEM [Chaudhry et al., ICLR 2019]: projects the gradient when it conflicts with the memory gradient
  gdumb    : GDumb [Prabhu et al., ECCV 2020]: greedy balanced memory, retrained from scratch on the memory
Upper reference:
  joint    : retrained on the data of all species seen so far
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from vcil.data import C

DEV = 'cuda'
REPLAY = ('er', 'icarl', 'der', 'agem', 'gdumb')
REG = ('ewc', 'si', 'mas')
BS = 256


class Head(nn.Module):
    def __init__(self, d_in, hidden=512, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.fc = nn.Linear(hidden, C)

    def forward(self, x, return_emb=False):
        e = self.body(x)
        return (self.fc(e), e) if return_emb else self.fc(e)


def kd_loss(logits, old_logits, T=2.0):
    return F.kl_div(F.log_softmax(logits / T, 1), F.softmax(old_logits / T, 1), reduction='batchmean') * T * T


class GradLearner:
    def __init__(self, method, d_in, hp, seed):
        torch.manual_seed(seed)
        self.method, self.hp, self.seed = method, hp, seed
        self.make = lambda: Head(d_in)
        self.model = self.make().to(DEV)
        self.old = None
        self.imp, self.star = None, None                         # regularisation: importance and anchor
        self.si_w = None
        self.mem_x = self.mem_y = self.mem_sp = self.mem_z = None   # replay memory (features, labels, species, logits)
        self.n_species = 0
        self.seen_x, self.seen_y = [], []                        # joint training only

    def _params(self):
        return list(self.model.parameters())

    def _mem_batch(self, g):
        idx = torch.randint(len(self.mem_x), (min(BS, len(self.mem_x)),), device=DEV, generator=g)
        return self.mem_x[idx], self.mem_y[idx], self.mem_z[idx]

    def _train(self, x, y, epochs, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        g = torch.Generator(device=DEV).manual_seed(self.seed * 100 + self.n_species)
        use_mem = self.mem_x is not None and self.method in ('der', 'agem')
        if self.method == 'si':
            start = [p.detach().clone() for p in self._params()]
            self.si_w = [torch.zeros_like(p) for p in self._params()]
        for _ in range(epochs):
            self.model.train()
            perm = torch.randperm(len(x), device=DEV, generator=g)
            for i in range(0, len(perm), BS):
                idx = perm[i:i + BS]
                xb, yb = x[idx], y[idx]
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                if self.old is not None and self.method in ('lwf', 'icarl'):
                    with torch.no_grad():
                        old_logits = self.old(xb)
                    loss = loss + self.hp['kd'] * kd_loss(logits, old_logits)
                if self.method in REG and self.imp is not None:
                    pen = sum((w * (p - s) ** 2).sum() for p, w, s in zip(self._params(), self.imp, self.star))
                    loss = loss + self.hp['reg'] / 2 * pen
                if self.method == 'der' and use_mem:
                    mx, _, mz = self._mem_batch(g)
                    loss = loss + self.hp['alpha'] * F.mse_loss(self.model(mx), mz)
                    mx, my, _ = self._mem_batch(g)
                    loss = loss + self.hp['beta'] * F.cross_entropy(self.model(mx), my)
                if self.method == 'agem' and use_mem:
                    mx, my, _ = self._mem_batch(g)
                    opt.zero_grad()
                    F.cross_entropy(self.model(mx), my).backward()
                    g_ref = torch.cat([p.grad.flatten() for p in self._params()])
                opt.zero_grad()
                loss.backward()
                if self.method == 'agem' and use_mem:
                    gr = torch.cat([p.grad.flatten() for p in self._params()])
                    dot = gr @ g_ref
                    if dot < 0:
                        gr = gr - dot / (g_ref @ g_ref) * g_ref
                        o = 0
                        for p in self._params():
                            p.grad.copy_(gr[o:o + p.numel()].view_as(p))
                            o += p.numel()
                if self.method == 'si':
                    before = [p.detach().clone() for p in self._params()]
                    grads = [p.grad.detach().clone() for p in self._params()]
                opt.step()
                if self.method == 'si':
                    for w, p, b, gd in zip(self.si_w, self._params(), before, grads):
                        w -= gd * (p.detach() - b)
        if self.method == 'si':
            xi = 0.1
            om = [w / ((p.detach() - s) ** 2 + xi) for w, p, s in zip(self.si_w, self._params(), start)]
            self.imp = om if self.imp is None else [a + b for a, b in zip(self.imp, om)]
            self.star = [p.detach().clone() for p in self._params()]

    def learn(self, x, y):
        k = self.n_species
        if self.method == 'joint':
            self.seen_x.append(x)
            self.seen_y.append(y)
            torch.manual_seed(self.seed + k)
            self.model = self.make().to(DEV)
            self._train(torch.cat(self.seen_x), torch.cat(self.seen_y), self.hp['epochs'], self.hp['lr'])
        elif self.method == 'gdumb':
            self._update_memory(x, y, k)
            torch.manual_seed(self.seed + k)
            self.model = self.make().to(DEV)
            self._train(self.mem_x, self.mem_y, self.hp['epochs'], self.hp['lr'])
        else:
            tx, ty = x, y
            if self.method in ('er', 'icarl') and self.mem_x is not None:
                tx, ty = torch.cat([x, self.mem_x]), torch.cat([y, self.mem_y])
            self._train(tx, ty, self.hp['epochs'], self.hp['lr'])
        if self.method in ('ewc', 'mas'):
            self._update_importance(x, y)
        if self.method in ('lwf', 'icarl'):
            self.old = copy.deepcopy(self.model).eval()
            for p_ in self.old.parameters():
                p_.requires_grad = False
        self.n_species += 1
        if self.method in REPLAY and self.method != 'gdumb':
            self._update_memory(x, y, k)

    def _update_importance(self, x, y):
        """EWC: diagonal empirical Fisher; MAS: |grad of the squared output L2 norm|. Both accumulated online."""
        self.model.eval()
        imp = [torch.zeros_like(p) for p in self._params()]
        n = 0
        for i in range(0, len(x), BS):
            self.model.zero_grad()
            out = self.model(x[i:i + BS])
            if self.method == 'ewc':
                F.cross_entropy(out, y[i:i + BS], reduction='sum').backward()
                for w, p in zip(imp, self._params()):
                    w += p.grad.detach() ** 2 / len(out)
            else:
                out.pow(2).sum(1).mean().backward()
                for w, p in zip(imp, self._params()):
                    w += p.grad.detach().abs()
            n += 1
        imp = [w / n for w in imp]
        self.imp = imp if self.imp is None else [a + b for a, b in zip(self.imp, imp)]
        self.star = [p.detach().clone() for p in self._params()]

    def _update_memory(self, x, y, k):
        """Fixed total budget K split equally over (species, level) cells; old cells are trimmed when a species arrives."""
        n_old = 0 if self.mem_x is None else len({(int(s), int(c)) for s, c in
                                                  zip(self.mem_sp.tolist(), self.mem_y.tolist())})
        per = self.hp['budget'] // max(1, n_old + int(torch.unique(y).numel()))
        xs, ys, sps, zs = [], [], [], []
        if self.mem_x is not None:
            for sp in range(k):
                for c in range(C):
                    m = torch.nonzero((self.mem_sp == sp) & (self.mem_y == c)).flatten()[:per]
                    xs.append(self.mem_x[m]); ys.append(self.mem_y[m]); sps.append(self.mem_sp[m]); zs.append(self.mem_z[m])
        g = torch.Generator(device=DEV).manual_seed(self.seed * 1000 + k)
        self.model.eval()
        for c in range(C):
            xc = x[y == c]
            if self.method != 'icarl':
                sel = torch.randperm(len(xc), device=DEV, generator=g)[:per]
            else:   # iCaRL herding: pick samples whose running mean is closest to the class mean
                with torch.no_grad():
                    e = F.normalize(self.model(xc, return_emb=True)[1], dim=1)
                mu = e.mean(0)
                sel, acc = [], torch.zeros_like(mu)
                avail = torch.ones(len(xc), dtype=torch.bool, device=DEV)
                for j in range(min(per, len(xc))):
                    d = ((mu[None] - (acc[None] + e) / (j + 1)) ** 2).sum(1)
                    d[~avail] = float('inf')
                    i = int(d.argmin())
                    sel.append(i); avail[i] = False; acc += e[i]
                sel = torch.tensor(sel, device=DEV, dtype=torch.long)
            with torch.no_grad():
                zs.append(self.model(xc[sel]))
            xs.append(xc[sel]); ys.append(torch.full((len(sel),), c, device=DEV, dtype=torch.long))
            sps.append(torch.full((len(sel),), k, device=DEV, dtype=torch.long))
        self.mem_x, self.mem_y, self.mem_sp, self.mem_z = torch.cat(xs), torch.cat(ys), torch.cat(sps), torch.cat(zs)

    @torch.no_grad()
    def predict(self, x, variant=None):
        self.model.eval()
        logits, e = self.model(x, return_emb=True)
        if variant != 'nme':
            return logits.argmax(1)
        em = F.normalize(self.model(self.mem_x, return_emb=True)[1], dim=1)
        means = torch.stack([F.normalize(em[self.mem_y == c].mean(0), dim=0) for c in range(C)])
        return (F.normalize(e, dim=1) @ means.T).argmax(1)
