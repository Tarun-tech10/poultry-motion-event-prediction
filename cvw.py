"""Grouped CV with family-importance weighting, scored on the BALANCED objective."""
import numpy as np, lightgbm as lgb, time, os, sys
from common import *
from feats3 import zone_features3
from feats2 import row_features
from scorer import score_arrays
from decode import decode
from evalbal import rule_families, balanced_eval, NAMES
from pipe import ZCFG, MCFG, RANK_W, RANK_T, RCFG, _p, TOPC

MODE = os.environ.get("WMODE", "bal")     # bal | mover | none
CAP = float(os.environ.get("CAP", 4.0))
TAG = os.environ.get("TAG", "w")

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
c2a = np.zeros((N, 16), np.int64); np.put_along_axis(c2a, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)
fold = np.load("fold_group.npy")
fam = rule_families(ZC, M)

if MODE == "none":
    W = np.ones(N)
elif MODE == "mover":
    CRc = np.array([[i // 4, i % 4] for i in range(16)])
    mv = np.abs(np.diff(CRc[ZC], axis=1)).sum(2).sum(1)
    W = 1.0 + 1.5 * (mv >= 2)
else:
    W = np.zeros(N)
    for c in range(5):
        s = fam == c
        W[s] = (1.0 / 5) / max(s.mean(), 1e-9)
    W = np.clip(W, 1.0 / CAP, CAP)
W = W / W.mean()
print(f"WMODE={MODE} CAP={CAP} weight range {W.min():.2f}-{W.max():.2f}", flush=True)
print("mean weight by family:", {NAMES[c]: round(float(W[fam == c].mean()), 2) for c in range(5)}, flush=True)

t0 = time.time()
XZ, _ = zone_features3(S, RC); XR, _ = row_features(S, RC)
F = XZ.shape[2]

def fit_zone_w(XZtr, ZCtr, Wtr, XZte, threads=14):
    ntr = XZtr.shape[0]; nte = XZte.shape[0]
    Ftr = XZtr.reshape(-1, F); Fte = XZte.reshape(-1, F)
    wf = np.repeat(Wtr, 16)
    out = np.zeros((nte, 3, 16), np.float32)
    for k in range(3):
        y = np.zeros((ntr, 16), np.float32); y[np.arange(ntr), ZCtr[:, k]] = 1
        acc = np.zeros((nte, 16))
        for cfg in ZCFG:
            m = lgb.train(_p(cfg, threads), lgb.Dataset(Ftr, y.reshape(-1), weight=wf), cfg["rounds"])
            acc += np.log(np.clip(m.predict(Fte).reshape(nte, 16), 1e-7, 1))
        acc /= len(ZCFG)
        rp = dict(_p(RCFG, threads), objective="lambdarank", lambdarank_truncation_level=4)
        rm = lgb.train(rp, lgb.Dataset(Ftr, y.reshape(-1), group=np.full(ntr, 16), weight=wf), RCFG["rounds"])
        rs = rm.predict(Fte).reshape(nte, 16) / RANK_T
        rs = rs - rs.max(1, keepdims=True)
        acc = (1 - RANK_W) * acc + RANK_W * (rs - np.log(np.exp(rs).sum(1, keepdims=True)))
        e = np.exp(acc - acc.max(1, keepdims=True))
        out[:, k] = (e / e.sum(1, keepdims=True)).astype(np.float32)
    return out

def fit_tier_w(XZtr, XRtr, Mtr, ZCtr, Wtr, XZte, XRte, pz, threads=14):
    nte = XZte.shape[0]; ntr = XZtr.shape[0]
    out = np.zeros((nte, 3, 4), np.float32)
    cand = np.argsort(-pz, axis=2)[:, :, :TOPC]
    for k in range(3):
        Xtr = np.concatenate([XRtr, XZtr[np.arange(ntr), ZCtr[:, k]]], 1)
        m = lgb.train(dict(MCFG, num_threads=threads, seed=7),
                      lgb.Dataset(Xtr, Mtr[:, k], weight=Wtr), 420)
        num = np.zeros((nte, 4)); den = np.zeros((nte, 1))
        for j in range(TOPC):
            c = cand[:, k, j]
            w = pz[np.arange(nte), k, c][:, None]
            num += w * m.predict(np.concatenate([XRte, XZte[np.arange(nte), c]], 1)); den += w
        out[:, k] = (num / np.maximum(den, 1e-9)).astype(np.float32)
    return out

oz = np.zeros((N, 3, 16), np.float32); om = np.zeros((N, 3, 4), np.float32)
for f in range(5):
    tr = fold != f; va = fold == f
    pz = fit_zone_w(XZ[tr], ZC[tr], W[tr], XZ[va])
    pm = fit_tier_w(XZ[tr], XR[tr], M[tr], ZC[tr], W[tr], XZ[va], XR[va], pz)
    oz[va] = pz; om[va] = pm
    print(f"fold {f} zone {[round(float((pz[:,k].argmax(1)==ZC[va,k]).mean()),4) for k in range(3)]} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{TAG}_oof_z.npy", oz); np.save(f"{TAG}_oof_m.npy", om)

od = np.zeros((N, 3, 9), np.float32); od[:, :, 8] = 1.0
famk = np.load("fam_target.npy")
best = None
for lam in [-0.5, 0.0, 0.5, 1.0, 1.5]:
    for T in [0.8, 0.9, 1.0, 1.15]:
        for Tm in [1.0, 1.15]:
            p = oz ** (1 / T); p /= p.sum(2, keepdims=True)
            q = om ** (1 / Tm); q /= q.sum(2, keepdims=True)
            cells, msel, dsel = decode(p, q, od, lam=lam, force_dc=True)
            rows = score_arrays(np.take_along_axis(c2a, cells, 1), dsel, msel, Z, D, M, RC, reduce=False)
            r1 = balanced_eval(rows, fam); r2 = balanced_eval(rows, famk)
            if best is None or r1[0] + r2[0] > best[0]:
                best = (r1[0] + r2[0], lam, T, Tm, r1, r2, rows)
_, lam, T, Tm, r1, r2, rows = best
print(f"\n[{MODE}] BEST lam={lam} T={T} Tm={Tm}", flush=True)
print(f"  BALANCED final {r1[0]:.4f} (ovr {r1[1]:.4f} worst {r1[2]:.4f} b20 {r1[3]:.4f}) | kmeans-fam {r2[0]:.4f} | train-mix ovr {rows.mean():.4f}")
print("  per-family:", {NAMES[i]: round(float(r1[4][i]), 4) for i in range(5)})
