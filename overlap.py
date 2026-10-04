"""Can another row's PAST cover this row's FUTURE?  (window-overlap exploit probe)"""
import numpy as np
from common import *
from scipy.optimize import linear_sum_assignment

d = load("../HEN_DATA/train.csv")
S, RC, Z, M = d["strip"], d["rc"], d["zone"], d["tier"]
N = len(Z)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
G = to_grid(S, RC).reshape(N, 16, 30)          # cell-indexed (transform still row-specific)

tr = S.sum(1)                                   # (N,30) permutation-invariant
def zs(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)

print("lag  L   #pairs(corr>.97)  #pairs(>.99)  best")
cands = {}
for lag in range(14, 26):
    L = 30 - lag
    if L < 5: break
    A = zs(tr[:, lag:lag + L]); B = zs(tr[:, :L])
    C = (A @ B.T) / L                            # C[i,j]: row j starts `lag` frames after row i
    np.fill_diagonal(C, -9)
    print(f"{lag:3d} {L:3d}  {(C>0.97).sum():8d} {(C>0.99).sum():12d}  {C.max():.4f}")
    cands[lag] = C

# ---- verify best candidates at lags that fully cover the future (18..29) ----
def verify(i, j, lag):
    """does row j's past really continue row i's past? match zones by Hungarian."""
    L = 30 - lag
    a = G[i, :, lag:lag + L]; b = G[j, :, :L]
    az = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8)
    bz = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-8)
    Cz = az @ bz.T / L                            # (16,16) zone-to-zone correlation
    r, c = linear_sum_assignment(-Cz)
    return Cz[r, c].mean(), c                     # c[i_zone] = matching zone in j

best_lag = 18
hits = []
for lag in [18, 19, 20, 21, 22]:
    if lag not in cands: continue
    C = cands[lag]
    ii, jj = np.where(C > 0.985)
    for i, j in list(zip(ii, jj))[:400]:
        q, perm = verify(i, j, lag)
        if q > 0.85: hits.append((i, j, lag, q, perm))
print("\nverified overlap pairs:", len(hits))
if hits:
    ok = 0; tot = 0
    for i, j, lag, q, perm in hits[:200]:
        # row i's future frames 30..47 == row j's frames (30-lag)..(47-lag)
        s = 30 - lag
        fut = G[j][:, s:s + 18]
        if fut.shape[1] < 18: continue
        # map j's cells back to i's cell indexing
        inv = np.zeros(16, int); inv[perm] = np.arange(16)
        for k in range(3):
            blk = fut[:, k * 6:(k + 1) * 6].mean(1)
            pred_in_j = blk.argmax()
            pred_in_i = inv[pred_in_j]
            ok += int(pred_in_i == ZC[i, k]); tot += 1
    print(f"derived-from-partner zone accuracy: {ok}/{tot} = {ok/max(tot,1):.4f}  (model gets ~0.47 avg)")
