import numpy as np, collections, sys
from common import *

tr = load("../HEN_DATA/train.csv"); te = load("../HEN_DATA/test.csv", False)
S = np.concatenate([tr["strip"], te["strip"]], 0)
istest = np.r_[np.zeros(len(tr["strip"]),bool), np.ones(len(te["strip"]),bool)]
N = len(S); print("N", N)

trace = S.sum(1)                      # N,30 permutation-invariant global motion trace
def zs(x, ax=-1):
    m = x.mean(ax, keepdims=True); s = x.std(ax, keepdims=True) + 1e-8
    return (x - m) / s

best = np.zeros((N, N), np.float32); bestlag = np.zeros((N, N), np.int8)
for d in range(0, 22):                # lag: row j starts d frames after row i
    L = 30 - d
    if L < 9: break
    A = zs(trace[:, d:d+L]); B = zs(trace[:, :L])
    C = (A @ B.T) / L
    m = C > best
    best[m] = C[m]; bestlag[m] = d
np.fill_diagonal(best, 0)
sym = np.maximum(best, best.T)
print("corr percentiles", np.percentile(sym, [50, 90, 99, 99.9]).round(3), "max", sym.max().round(4))
for thr in [0.90, 0.95, 0.97, 0.99]:
    print(f" thr {thr}: pairs {(sym>thr).sum()//2}, rows with >=1 partner {((sym>thr).sum(1)>0).mean():.3f}")

# connected components at a threshold
def comps(thr):
    adj = sym > thr
    lab = -np.ones(N, int); c = 0
    for i in range(N):
        if lab[i] >= 0: continue
        stack = [i]; lab[i] = c
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                if lab[v] < 0: lab[v] = c; stack.append(v)
        c += 1
    return lab, c

for thr in [0.97, 0.95, 0.93, 0.90]:
    lab, c = comps(thr)
    sizes = np.bincount(lab)
    big = np.sort(sizes)[::-1][:25]
    mixed = sum(1 for g in range(c) if istest[lab==g].any() and (~istest[lab==g]).any())
    print(f"thr {thr}: {c} comps, top sizes {big[:12].tolist()}, singletons {(sizes==1).sum()}, train/test-mixed comps {mixed}")
    np.save(f"comp_{thr}.npy", lab)
