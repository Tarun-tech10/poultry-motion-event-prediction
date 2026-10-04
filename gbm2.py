import numpy as np, lightgbm as lgb, time, sys, os
from common import *
from feats3 import zone_features3 as zone_features2
from feats2 import row_features
from scorer import score_arrays
from decode import decode
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

CHAIN = int(os.environ.get("CHAIN", 1))
NROUND = int(os.environ.get("NROUND", 1600))
LR = float(os.environ.get("LR", 0.03))
RANDCV = int(os.environ.get("RANDCV", 0))

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
cell2alias = np.zeros((N, 16), np.int64)
np.put_along_axis(cell2alias, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)

t0 = time.time()
XZ, znames = zone_features2(S, RC); XR, rnames = row_features(S, RC)
print("zone feats", XZ.shape, f"({time.time()-t0:.0f}s)", flush=True)

def sig(S):
    f = [np.sort(S.mean(2), 1), S.sum(1),
         np.stack([S.mean((1,2)), S.std((1,2)), S.max((1,2)), np.diff(S,axis=2).std((1,2)), S.mean(2).std(1)], 1)]
    return np.concatenate([(x - x.mean(0)) / (x.std(0) + 1e-8) for x in f], 1)
FOLDS = 5
if RANDCV:
    fold = np.zeros(N, int)
    for i, (_, v) in enumerate(KFold(FOLDS, shuffle=True, random_state=0).split(np.arange(N))): fold[v] = i
else:
    grp = KMeans(14, n_init=8, random_state=0).fit_predict(sig(S))
    loadv = np.zeros(FOLDS); fold_of = {}
    for g in np.argsort(-np.bincount(grp)):
        fq = loadv.argmin(); fold_of[g] = fq; loadv[fq] += (grp == g).sum()
    fold = np.load("fold_group.npy")
np.save("ZC.npy", ZC); np.save("cell_of.npy", cell_of)
print("fold sizes", np.bincount(fold), "RANDCV", RANDCV, flush=True)

PZ = dict(objective="binary", learning_rate=LR, num_leaves=63, min_data_in_leaf=80,
          feature_fraction=0.45, bagging_fraction=0.8, bagging_freq=1, lambda_l2=8.0,
          verbose=-1, num_threads=10, max_bin=127)

def chain_feats(prev_p):
    """features derived from the previous horizon's OOF distribution: (N,16)->(N,16,4)"""
    top = prev_p.argmax(1)
    dist = np.abs(CRc[None, :, :] - CRc[top][:, None, :]).sum(-1).astype(np.float32)
    nb = prev_p @ A1n.T
    return np.stack([prev_p, nb, dist, np.broadcast_to(prev_p.max(1)[:, None], prev_p.shape)], -1)
from feats import CR as CRc, A1 as A1n

oof_z = np.zeros((N, 3, 16), np.float32)
best_it = []
for k in range(3):
    Xk = XZ if (k == 0 or not CHAIN) else np.concatenate([XZ, chain_feats(oof_z[:, k-1])], -1)
    nm = znames if (k == 0 or not CHAIN) else znames + [f"ch_{i}" for i in range(4)]
    Xf = Xk.reshape(N * 16, -1)
    y = np.zeros((N, 16), np.float32); y[np.arange(N), ZC[:, k]] = 1; yf = y.reshape(-1)
    for f in range(FOLDS):
        tri = np.repeat(fold != f, 16); vai = np.repeat(fold == f, 16)
        ds = lgb.Dataset(Xf[tri], yf[tri], feature_name=nm)
        dv = lgb.Dataset(Xf[vai], yf[vai], reference=ds)
        m = lgb.train(PZ, ds, NROUND, valid_sets=[dv], callbacks=[lgb.early_stopping(80, verbose=False)])
        best_it.append(m.best_iteration)
        oof_z[fold == f, k] = m.predict(Xf[vai], num_iteration=m.best_iteration).reshape(-1, 16)
    lo = np.log(np.clip(oof_z[:, k], 1e-7, 1))
    oof_z[:, k] = np.exp(lo - lo.max(1, keepdims=True)); oof_z[:, k] /= oof_z[:, k].sum(1, keepdims=True)
    print(f"H{k} zone acc {(oof_z[:,k].argmax(1)==ZC[:,k]).mean():.4f} best_it {best_it[-FOLDS:]} ({time.time()-t0:.0f}s)", flush=True)

# tier
oof_m = np.zeros((N, 3, 4), np.float32)
PM = dict(objective="multiclass", num_class=4, learning_rate=0.035, num_leaves=31, min_data_in_leaf=60,
          feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=10)
for k in range(3):
    # expected zone features under the predicted distribution + best-cell features
    exp_f = (XZ * oof_z[:, k][:, :, None]).sum(1)
    bc = oof_z[:, k].argmax(1)
    best_f = XZ[np.arange(N), bc]
    Xm = np.concatenate([XR, exp_f, best_f, oof_z[:, k].max(1, keepdims=True)], 1)
    for f in range(FOLDS):
        tri = fold != f; vai = fold == f
        ds = lgb.Dataset(Xm[tri], M[tri, k]); dv = lgb.Dataset(Xm[vai], M[vai, k], reference=ds)
        m = lgb.train(PM, ds, 800, valid_sets=[dv], callbacks=[lgb.early_stopping(60, verbose=False)])
        oof_m[vai, k] = m.predict(Xm[vai], num_iteration=m.best_iteration)
    print(f"H{k} tier acc {(oof_m[:,k].argmax(1)==M[:,k]).mean():.4f}", flush=True)

tag = os.environ.get("TAG", "v2")
np.save(f"{tag}_oof_z.npy", oof_z); np.save(f"{tag}_oof_m.npy", oof_m)
oof_d = np.zeros((N, 3, 9), np.float32); oof_d[:, :, 8] = 1.0
for lam in [0.0, 0.5, 1.0, 2.0]:
    cells, msel, dsel = decode(oof_z, oof_m, oof_d, lam=lam, force_dc=True)
    al = np.take_along_axis(cell2alias, cells, 1)
    r = score_arrays(al, dsel, msel, Z, D, M, RC)
    print(f"lam {lam}: final {r['final']:.4f} ovr {r['overall']:.4f} zone {r['zone']:.3f} tier {r['tier']:.3f} trans {r['trans']:.3f} b20 {r['bot20']:.3f}", flush=True)
