"""Is there ANY recoverable signal about WHERE motion moves?  (ceiling probe for crowd_shift)"""
import numpy as np, lightgbm as lgb
from common import *
from feats3 import zone_features3
from feats2 import row_features
from evalbal import rule_families
from sklearn.metrics import roc_auc_score

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC).reshape(N, 16, 30)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
CRc = np.array([[i // 4, i % 4] for i in range(16)])
fold = np.load("fold_group.npy")
fam = rule_families(ZC, M)
hot = G[:, :, -3:].mean(2).argmax(1)

# ---- 1. does estimated optical flow point at the actual displacement? ----
G4 = G.reshape(N, 4, 4, 30)
SH = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], np.float32)
def flow(lag):
    a = G4[:, :, :, :30 - lag]; b = G4[:, :, :, lag:]
    a = a - a.mean((1, 2), keepdims=True); b = b - b.mean((1, 2), keepdims=True)
    sc = np.zeros((N, 9), np.float32)
    for i, (dr, dc) in enumerate(SH):
        dr, dc = int(dr), int(dc)
        r0, r1 = max(0, -dr), 4 - max(0, dr); c0, c1 = max(0, -dc), 4 - max(0, dc)
        aa = a[:, r0:r1, c0:c1]; bb = b[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        sc[:, i] = (aa * bb).sum((1, 2, 3)) / (aa.shape[1] * aa.shape[2])
    sc -= sc.min(1, keepdims=True); w = sc / (sc.sum(1, keepdims=True) + 1e-6)
    return w @ SH / lag
v1, v3 = flow(1), flow(3)
disp = (CRc[ZC[:, 2]] - CRc[hot]).astype(np.float32)      # actual H2 displacement from hot cell
mvr = fam == 0
for nm, v in [("flow lag1", v1), ("flow lag3", v3)]:
    for sel, lbl in [(np.ones(N, bool), "all"), (mvr, "crowd_shift")]:
        dn = np.linalg.norm(disp[sel], axis=1) + 1e-6; vn = np.linalg.norm(v[sel], axis=1) + 1e-6
        cos = (disp[sel] * v[sel]).sum(1) / (dn * vn)
        print(f"{nm:10s} {lbl:12s} cos(flow, actual displacement) = {cos.mean():+.4f} (0 = no signal)")

# ---- 2. supervised ceiling: can a model trained ONLY on movers find the destination? ----
XZ, _ = zone_features3(S, RC); XR, _ = row_features(S, RC)
F = XZ.shape[2]
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
         feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
         verbose=-1, num_threads=8, max_bin=127)
print("\n--- zone model trained ONLY on mover rows, evaluated on mover rows ---")
for k in [1, 2]:
    acc_s = []; acc_g = []
    for f in range(5):
        tr = (fold != f) & mvr; va = (fold == f) & mvr
        if va.sum() < 5: continue
        ntr = tr.sum()
        y = np.zeros((ntr, 16), np.float32); y[np.arange(ntr), ZC[tr, k]] = 1
        m = lgb.train(P, lgb.Dataset(XZ[tr].reshape(-1, F), y.reshape(-1)), 250)
        p = m.predict(XZ[va].reshape(-1, F)).reshape(-1, 16)
        acc_s.append((p.argmax(1) == ZC[va, k]).mean())
        # general model for comparison
        trg = fold != f; ntrg = trg.sum()
        yg = np.zeros((ntrg, 16), np.float32); yg[np.arange(ntrg), ZC[trg, k]] = 1
        mg = lgb.train(P, lgb.Dataset(XZ[trg].reshape(-1, F), yg.reshape(-1)), 250)
        pg = mg.predict(XZ[va].reshape(-1, F)).reshape(-1, 16)
        acc_g.append((pg.argmax(1) == ZC[va, k]).mean())
    print(f"  H{k}: mover-specialist {np.mean(acc_s):.4f}   general model {np.mean(acc_g):.4f}   chance 0.0625")

# ---- 3. can we even DETECT movers in advance? ----
Xrow = np.concatenate([XR, XZ[np.arange(N), hot]], 1)
o = np.zeros(N)
for f in range(5):
    tr = fold != f; va = fold == f
    m = lgb.train(dict(P, num_threads=12), lgb.Dataset(Xrow[tr], mvr[tr].astype(int)), 300)
    o[va] = m.predict(Xrow[va])
print(f"\ncrowd_shift detectability: base rate {mvr.mean():.3f}  AUC {roc_auc_score(mvr, o):.4f}"
      f"  top-decile precision {mvr[o >= np.quantile(o, .9)].mean():.3f}")
