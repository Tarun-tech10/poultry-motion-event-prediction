"""GBM + multi-seed GNN ensemble, tuned on the BALANCED objective."""
import numpy as np, glob, itertools, os, json
from common import *
from scorer import score_arrays
from decode import decode
from evalbal import rule_families, balanced_eval, NAMES

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
c2a = np.zeros((N, 16), np.int64); np.put_along_axis(c2a, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)
fam = rule_families(ZC, M); famk = np.load("fam_target.npy")
od = np.zeros((N, 3, 9), np.float32); od[:, :, 8] = 1.0
ez = np.load("ens_oof_z.npy"); em = np.load("ens_oof_m.npy")

SEEDS = [s for s in os.environ.get("SEEDS", "1,5,6,7").split(",") if os.path.exists(f"gnn{s}_oof_z.npy")]
print("GNN seeds used:", SEEDS)
gz = np.mean([np.log(np.clip(np.load(f"gnn{s}_oof_z.npy"), 1e-7, 1)) for s in SEEDS], 0)
e = np.exp(gz - gz.max(2, keepdims=True)); gz = (e / e.sum(2, keepdims=True)).astype(np.float32)
gm = np.mean([np.load(f"gnn{s}_oof_m.npy") for s in SEEDS], 0).astype(np.float32)
print("GNN-ens zone acc", [round(float((gz[:, k].argmax(1) == ZC[:, k]).mean()), 4) for k in range(3)])
print("GBM     zone acc", [round(float((ez[:, k].argmax(1) == ZC[:, k]).mean()), 4) for k in range(3)])

def ev(pz, pm, lam, T, Tm):
    p = pz ** (1 / T); p /= p.sum(2, keepdims=True)
    q = pm ** (1 / Tm); q /= q.sum(2, keepdims=True)
    cells, msel, dsel = decode(p, q, od, lam=lam, force_dc=True)
    rows = score_arrays(np.take_along_axis(c2a, cells, 1), dsel, msel, Z, D, M, RC, reduce=False)
    return balanced_eval(rows, fam), balanced_eval(rows, famk), rows.mean()

best = None
for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]:
    lo = (1 - w) * np.log(np.clip(ez, 1e-7, 1)) + w * np.log(np.clip(gz, 1e-7, 1))
    x = np.exp(lo - lo.max(2, keepdims=True)); bz = (x / x.sum(2, keepdims=True)).astype(np.float32)
    for wm in [0.0, 0.3, 0.5]:
        bm = ((1 - wm) * em + wm * gm).astype(np.float32)
        for lam in [1.0, 1.5, 2.0]:
            for T in [0.8, 0.9, 1.0, 1.15]:
                for Tm in [1.0, 1.15]:
                    r1, r2, mix = ev(bz, bm, lam, T, Tm)
                    sc = r1[0] + r2[0]
                    if best is None or sc > best[0]:
                        best = (sc, w, wm, lam, T, Tm, r1, r2, mix)
_, w, wm, lam, T, Tm, r1, r2, mix = best
print(f"\nBEST blend w={w} wm={wm} lam={lam} T={T} Tm={Tm}")
print(f"  BALANCED final {r1[0]:.4f} (ovr {r1[1]:.4f} worst {r1[2]:.4f} b20 {r1[3]:.4f}) | kmeans-fam {r2[0]:.4f} | train-mix ovr {mix:.4f}")
print("  per-family:", {NAMES[i]: round(float(r1[4][i]), 4) for i in range(5)})
r1b, _, _ = ev(ez, em, 1.5, 0.9, 1.0)
print(f"  (shipped v1 baseline: balanced {r1b[0]:.4f} worst {r1b[2]:.4f})")
json.dump(dict(w=w, wm=wm, lam=lam, T=T, Tm=Tm, seeds=SEEDS), open("final_params.json", "w"))
print("saved final_params.json")
