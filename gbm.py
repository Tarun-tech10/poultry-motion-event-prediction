import numpy as np, lightgbm as lgb, time, sys, os, json
from common import *
from feats import zone_features, row_features
from scorer import score_arrays
from decode import decode
from sklearn.cluster import KMeans

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))     # alias -> cell
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)                          # target cell
cell2alias = np.zeros((N, 16), np.int64)
np.put_along_axis(cell2alias, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)

t0 = time.time()
XZ, znames = zone_features(S, RC); XR, rnames = row_features(S, RC)
print("zone feats", XZ.shape, "row feats", XR.shape, f"({time.time()-t0:.0f}s)", flush=True)

# ---- group CV (source-proxy clusters) ----
def sig(S):
    f = [np.sort(S.mean(2), 1), S.sum(1),
         np.stack([S.mean((1,2)), S.std((1,2)), S.max((1,2)), np.diff(S,axis=2).std((1,2)), S.mean(2).std(1)], 1)]
    return np.concatenate([(x - x.mean(0)) / (x.std(0) + 1e-8) for x in f], 1)
grp = KMeans(14, n_init=8, random_state=0).fit_predict(sig(S))
FOLDS = 5; loadv = np.zeros(FOLDS); fold_of = {}
for g in np.argsort(-np.bincount(grp)):
    fq = loadv.argmin(); fold_of[g] = fq; loadv[fq] += (grp == g).sum()
fold = np.array([fold_of[g] for g in grp])
np.save("fold.npy", fold); np.save("ZC.npy", ZC); np.save("cell_of.npy", cell_of)
print("fold sizes", np.bincount(fold), flush=True)

# ---- zone model: per-cell binary relevance, softmax-normalised ----
Xflat = XZ.reshape(N * 16, -1)
oof_z = np.zeros((N, 3, 16), np.float32)
PZ = dict(objective="binary", learning_rate=0.045, num_leaves=63, min_data_in_leaf=80,
          feature_fraction=0.55, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
          verbose=-1, num_threads=16, max_bin=127)
NROUND = int(os.environ.get("NROUND", 700))
models_z = []
for k in range(3):
    y = np.zeros((N, 16), np.float32); y[np.arange(N), ZC[:, k]] = 1
    yf = y.reshape(-1)
    for f in range(FOLDS):
        tri = np.repeat(fold != f, 16); vai = np.repeat(fold == f, 16)
        ds = lgb.Dataset(Xflat[tri], yf[tri], feature_name=znames)
        dv = lgb.Dataset(Xflat[vai], yf[vai], reference=ds)
        m = lgb.train(PZ, ds, NROUND, valid_sets=[dv], callbacks=[lgb.early_stopping(60, verbose=False)])
        p = m.predict(Xflat[vai], num_iteration=m.best_iteration).reshape(-1, 16)
        oof_z[fold == f, k] = p
        models_z.append((k, f, m))
    lo = np.log(np.clip(oof_z[:, k], 1e-7, 1))
    oof_z[:, k] = np.exp(lo - lo.max(1, keepdims=True)); oof_z[:, k] /= oof_z[:, k].sum(1, keepdims=True)
    print(f"H{k} zone acc {(oof_z[:,k].argmax(1)==ZC[:,k]).mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- tier model: row-level multiclass, with top-zone context ----
oof_m = np.zeros((N, 3, 4), np.float32)
best_cell = oof_z.argmax(2)
extra = np.stack([np.take_along_axis(XZ, best_cell[:, k][:, None, None].repeat(XZ.shape[2], 2), 1)[:, 0, :] for k in range(3)], 1)
PM = dict(objective="multiclass", num_class=4, learning_rate=0.04, num_leaves=31, min_data_in_leaf=60,
          feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=16)
for k in range(3):
    Xm = np.concatenate([XR, extra[:, k]], 1)
    for f in range(FOLDS):
        tri = fold != f; vai = fold == f
        ds = lgb.Dataset(Xm[tri], M[tri, k]); dv = lgb.Dataset(Xm[vai], M[vai, k], reference=ds)
        m = lgb.train(PM, ds, 500, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_m[vai, k] = m.predict(Xm[vai], num_iteration=m.best_iteration)
    print(f"H{k} tier acc {(oof_m[:,k].argmax(1)==M[:,k]).mean():.4f}", flush=True)

oof_d = np.zeros((N, 3, 9), np.float32); oof_d[:, :, 8] = 1.0
np.save("g_oof_z.npy", oof_z); np.save("g_oof_m.npy", oof_m)

def ev(lam, lbl=""):
    cells, msel, dsel = decode(oof_z, oof_m, oof_d, lam=lam, force_dc=True)
    al = np.take_along_axis(cell2alias, cells, 1)
    r = score_arrays(al, dsel, msel, Z, D, M, RC)
    print(f"{lbl}lam {lam}: final {r['final']:.4f} ovr {r['overall']:.4f} zone {r['zone']:.3f} tier {r['tier']:.3f} trans {r['trans']:.3f} b20 {r['bot20']:.3f}", flush=True)
    return r['final']
for lam in [0.0, 0.5, 1.0, 2.0, 4.0]: ev(lam)
cells = oof_z.argmax(2); al = np.take_along_axis(cell2alias, cells, 1)
r = score_arrays(al, np.full((N,3),8), oof_m.argmax(2), Z, D, M, RC)
print(f"plain argmax: final {r['final']:.4f} ovr {r['overall']:.4f} zone {r['zone']:.3f} tier {r['tier']:.3f}", flush=True)
