import numpy as np, time, os, sys
from common import *
from feats3 import zone_features3
from feats2 import row_features
from scorer import score_arrays
from decode import decode
from pipe import fit_zone, fit_tier
from sklearn.cluster import KMeans

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
cell2alias = np.zeros((N, 16), np.int64)
np.put_along_axis(cell2alias, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)
fold = np.load("fold_group.npy")

t0 = time.time()
XZ, _ = zone_features3(S, RC); XR, _ = row_features(S, RC)
print("feats", XZ.shape, f"{time.time()-t0:.0f}s", flush=True)

oz = np.zeros((N, 3, 16), np.float32); om = np.zeros((N, 3, 4), np.float32)
for f in range(5):
    tr = fold != f; va = fold == f
    pz = fit_zone(XZ[tr], ZC[tr], XZ[va], threads=14)
    pm = fit_tier(XZ[tr], XR[tr], M[tr], ZC[tr], XZ[va], XR[va], pz, threads=14)
    oz[va] = pz; om[va] = pm
    print(f"fold {f} zone {[round(float((pz[:,k].argmax(1)==ZC[va,k]).mean()),4) for k in range(3)]} "
          f"tier {[round(float((pm[:,k].argmax(1)==M[va,k]).mean()),4) for k in range(3)]} ({time.time()-t0:.0f}s)", flush=True)
np.save("ens_oof_z.npy", oz); np.save("ens_oof_m.npy", om)
print("zone acc", [round(float((oz[:,k].argmax(1)==ZC[:,k]).mean()),4) for k in range(3)], flush=True)
print("tier acc", [round(float((om[:,k].argmax(1)==M[:,k]).mean()),4) for k in range(3)], flush=True)

# family-aware tuning
def scen(S, G):
    zm = S.mean(2); srt = np.sort(zm,1)[:,::-1]
    em = np.zeros((4,4),bool); em[0]=em[3]=True; em[:,0]=em[:,3]=True
    gm = G.mean(3); rr=np.arange(4)[None,:,None,None]; cc=np.arange(4)[None,None,:,None]
    w=G+1e-6; tot=w.sum((1,2)); cr=(w*rr).sum((1,2))/tot; cc_=(w*cc).sum((1,2))/tot
    t=np.arange(30)-14.5; vr=(cr*t).sum(1)/(t**2).sum(); vc=(cc_*t).sum(1)/(t**2).sum()
    return np.stack([srt[:,0]/(zm.mean(1)+1e-6), srt[:,1]/(srt[:,0]+1e-6), S.mean((1,2)),
        S[:,:,-10:].mean((1,2))-S[:,:,:10].mean((1,2)),
        S[:,:,-10:].mean((1,2))/(S[:,:,:10].mean((1,2))+1e-6),
        gm[:,em].mean(1)/(gm.mean((1,2))+1e-6), np.hypot(vr,vc),
        np.hypot(np.diff(cr,axis=1),np.diff(cc_,axis=1)).mean(1)],1)
G = to_grid(S, RC); Fz = scen(S,G); Fz=(Fz-Fz.mean(0))/(Fz.std(0)+1e-8)
fam = KMeans(5, n_init=10, random_state=0).fit_predict(Fz)
od = np.zeros((N,3,9), np.float32); od[:,:,8]=1.0

def full(lam, T, Tm=1.0):
    p = oz**(1.0/T); p/=p.sum(2,keepdims=True)
    q = om**(1.0/Tm); q/=q.sum(2,keepdims=True)
    cells, msel, dsel = decode(p, q, od, lam=lam, force_dc=True)
    al = np.take_along_axis(cell2alias, cells, 1)
    rows = score_arrays(al, dsel, msel, Z, D, M, RC, reduce=False)
    fm = np.array([rows[fam==c].mean() for c in range(5)])
    b20 = np.sort(rows)[:int(0.2*N)].mean()
    return 0.70*rows.mean()+0.18*fm.min()+0.12*b20, rows.mean(), fm.min(), b20
best=None
for lam in [1.0,1.5,2.0]:
    for T in [0.8,0.9,1.0]:
        for Tm in [1.0,1.15]:
            r=full(lam,T,Tm)
            if best is None or r[0]>best[0]: best=(r[0],lam,T,Tm,r[1],r[2],r[3])
print(f"BEST final {best[0]:.4f}  lam {best[1]} T {best[2]} Tm {best[3]} | overall {best[4]:.4f} worst {best[5]:.4f} b20 {best[6]:.4f}", flush=True)
print("params ->", dict(lam=best[1], T=best[2], Tm=best[3]))
