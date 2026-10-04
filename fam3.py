"""Rule-based families matching the 5 published names + how predictable 'stays put' is."""
import numpy as np, lightgbm as lgb
from common import *
from scorer import score_arrays
from decode import decode
from feats3 import zone_features3
from feats2 import row_features
from sklearn.metrics import roc_auc_score

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC).reshape(N, 16, 30)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
c2a = np.zeros((N, 16), np.int64); np.put_along_axis(c2a, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)
CRc = np.array([[i // 4, i % 4] for i in range(16)])
fold = np.load("fold_group.npy")

pos = CRc[ZC]
span = np.abs(pos[:, 2] - pos[:, 0]).sum(1)
mv = np.abs(np.diff(pos, axis=1)).sum(2).sum(1)          # total path length
ndist = np.array([len(set(z)) for z in ZC])
isedge = ((pos[:, :, 0] == 0) | (pos[:, :, 0] == 3) | (pos[:, :, 1] == 0) | (pos[:, :, 1] == 3)).mean(1)
tmean = M.mean(1)
ret = (ZC[:, 0] == ZC[:, 2]) & (ZC[:, 0] != ZC[:, 1])    # A-B-A alternation

# rule-based, mutually exclusive, priority order
fam = np.full(N, 4, np.int64)                             # 4 = steady_flow (default)
fam[(tmean <= 1.0)] = 2                                   # quiet_break
fam[(fam == 4) & ret] = 3                                 # multi_peak (alternating)
fam[(fam == 4) & (mv >= 3)] = 0                           # crowd_shift (long path)
fam[(fam == 4) & (isedge >= 0.99) & (tmean >= 2.0)] = 1   # edge_surge
NAMES = ["crowd_shift", "edge_surge", "quiet_break", "multi_peak", "steady_flow"]
print("rule-based family sizes:", {NAMES[i]: int((fam == i).sum()) for i in range(5)})

oz = np.load("ens_oof_z.npy"); om = np.load("ens_oof_m.npy")
od = np.zeros((N, 3, 9), np.float32); od[:, :, 8] = 1.0
p = oz ** (1 / 0.9); p /= p.sum(2, keepdims=True)
cells, msel, dsel = decode(p, om, od, lam=1.5, force_dc=True)
rows = score_arrays(np.take_along_axis(c2a, cells, 1), dsel, msel, Z, D, M, RC, reduce=False)
print(f"\noverall {rows.mean():.4f}")
fm = {}
for c in range(5):
    s = fam == c
    fm[NAMES[c]] = rows[s].mean()
    print(f"  {NAMES[c]:12s} n={s.sum():4d} ({s.mean()*100:4.1f}%) score {rows[s].mean():.4f} "
          f"zoneacc {[round(float((cells[s,k]==ZC[s,k]).mean()),3) for k in range(3)]}")
print("worst rule-family:", round(min(fm.values()), 4))
bal = np.mean([rows[fam == c].mean() for c in range(5)])
print(f"BALANCED overall (equal family weight, as the hidden set is): {bal:.4f}  vs train-mix {rows.mean():.4f}")
b20 = np.sort(rows)[:int(.2 * N)].mean()
print(f"=> implied final under balanced mix: {0.70*bal + 0.18*min(fm.values()) + 0.12*b20:.4f}")
np.save("fam_rule.npy", fam)

# ---------- is 'stays put' predictable from the past? ----------
XZ, _ = zone_features3(S, RC); XR, _ = row_features(S, RC)
hot = G[:, :, -3:].mean(2).argmax(1)
Xrow = np.concatenate([XR, XZ[np.arange(N), hot], oz.reshape(N, -1)], 1)
print("\n--- P(stay) models (grouped CV AUC) ---")
targets = {"H0==hot": (ZC[:, 0] == hot).astype(int),
           "H1==H0":  (ZC[:, 1] == ZC[:, 0]).astype(int),
           "H2==H1":  (ZC[:, 2] == ZC[:, 1]).astype(int)}
P = dict(objective="binary", learning_rate=0.04, num_leaves=31, min_data_in_leaf=60,
         feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
         verbose=-1, num_threads=12, max_bin=127)
stay = {}
for nm, y in targets.items():
    o = np.zeros(N)
    for f in range(5):
        tr = fold != f; va = fold == f
        m = lgb.train(P, lgb.Dataset(Xrow[tr], y[tr]), 300)
        o[va] = m.predict(Xrow[va])
    stay[nm] = o
    print(f"  {nm:8s} base rate {y.mean():.3f}  AUC {roc_auc_score(y, o):.4f}  "
          f"[decile1 {y[o<=np.quantile(o,.1)].mean():.3f} .. decile10 {y[o>=np.quantile(o,.9)].mean():.3f}]")
np.save("stay_oof.npy", np.stack([stay["H1==H0"], stay["H2==H1"]], 1))
