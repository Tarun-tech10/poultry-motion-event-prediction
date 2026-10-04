"""Recover the real scenario families from TARGET behaviour (families describe future motion)."""
import numpy as np, collections
from common import *
from scorer import score_arrays
from decode import decode
from sklearn.cluster import KMeans

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
c2a = np.zeros((N, 16), np.int64); np.put_along_axis(c2a, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)
CRc = np.array([[i // 4, i % 4] for i in range(16)])

# ---- target-side descriptors (what the FUTURE does) ----
pos = CRc[ZC]                                     # (N,3,2)
step1 = np.abs(pos[:, 1] - pos[:, 0]).sum(1)
step2 = np.abs(pos[:, 2] - pos[:, 1]).sum(1)
span = np.abs(pos[:, 2] - pos[:, 0]).sum(1)
ndist = np.array([len(set(z)) for z in ZC])
edge = ((pos[:, :, 0] == 0) | (pos[:, :, 0] == 3) | (pos[:, :, 1] == 0) | (pos[:, :, 1] == 3)).mean(1)
hot = G.reshape(N, 16, 30)[:, :, -3:].mean(2).argmax(1)
d_hot = np.abs(CRc[ZC] - CRc[hot][:, None, :]).sum(2)     # (N,3) distance from current hot cell
tier_mean = M.mean(1); tier_rng = M.max(1) - M.min(1); tier_tr = M[:, 2] - M[:, 0]
dc = (D == 8).mean(1)
T = np.stack([tier_mean, tier_rng, tier_tr, step1, step2, span, ndist, edge,
              d_hot[:, 0], d_hot[:, 1], d_hot[:, 2], dc], 1).astype(np.float32)
Tz = (T - T.mean(0)) / (T.std(0) + 1e-8)

print("=== KMeans on TARGET descriptors ===")
for k in [5]:
    km = KMeans(k, n_init=25, random_state=0).fit(Tz)
    fam = km.labels_
    print("sizes", np.bincount(fam), " (balanced would be ~%d)" % (N // k))
    names = ["tier_mean","tier_rng","tier_tr","step1","step2","span","ndist","edge","dhot0","dhot1","dhot2","dcfrac"]
    print("      " + " ".join(f"{n:>8}" for n in names))
    for c in range(k):
        print(f" c{c}  " + " ".join(f"{v:8.2f}" for v in T[fam == c].mean(0)))
np.save("fam_target.npy", fam)

# ---- how does the current model score on each target-family? ----
oz = np.load("ens_oof_z.npy"); om = np.load("ens_oof_m.npy")
od = np.zeros((N, 3, 9), np.float32); od[:, :, 8] = 1.0
p = oz ** (1 / 0.9); p /= p.sum(2, keepdims=True)
cells, msel, dsel = decode(p, om, od, lam=1.5, force_dc=True)
al = np.take_along_axis(c2a, cells, 1)
rows = score_arrays(al, dsel, msel, Z, D, M, RC, reduce=False)
print(f"\nOVERALL {rows.mean():.4f}   bottom20 {np.sort(rows)[:int(.2*N)].mean():.4f}")
print("score by TARGET-family:")
for c in range(5):
    s = fam == c
    print(f"  c{c} n={s.sum():4d} score {rows[s].mean():.4f}  zoneacc {[round(float((cells[s,k]==ZC[s,k]).mean()),3) for k in range(3)]}"
          f"  tieracc {[round(float((msel[s,k]==M[s,k]).mean()),3) for k in range(3)]}")
print("worst target-family:", round(min(rows[fam==c].mean() for c in range(5)), 4))

# ---- what distinguishes the bottom 20%? ----
thr = np.sort(rows)[int(.2 * N)]
bad = rows <= thr
print(f"\nbottom-20% profile (n={bad.sum()}):")
for nm, v in [("tier_mean", tier_mean), ("step1", step1), ("step2", step2), ("span", span),
              ("ndist", ndist), ("edge", edge), ("d_hot_H0", d_hot[:, 0]), ("d_hot_H2", d_hot[:, 2]),
              ("past_level", S.mean((1, 2))), ("peakiness", np.sort(S.mean(2), 1)[:, -1] / S.mean((1, 2)))]:
    print(f"  {nm:11s} bad {v[bad].mean():7.3f}   good {v[~bad].mean():7.3f}")
print("\ntier dist in bottom20:", collections.Counter(M[bad].ravel().tolist()).most_common())
print("tier dist overall     :", collections.Counter(M.ravel().tolist()).most_common())
