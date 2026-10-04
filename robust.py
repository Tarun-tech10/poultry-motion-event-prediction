import numpy as np, sys, os
from common import *
from scorer import score_arrays
from decode import decode
from sklearn.cluster import KMeans

TAG = os.environ.get("TAG", "v2")
d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC)
cell_of = (RC[:,:,0].astype(np.int64)*4 + RC[:,:,1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
cell2alias = np.zeros((N,16), np.int64)
np.put_along_axis(cell2alias, cell_of, np.arange(16)[None,:].repeat(N,0), 1)

oz = np.load(f"{TAG}_oof_z.npy"); om = np.load(f"{TAG}_oof_m.npy")
od = np.zeros((N,3,9), np.float32); od[:,:,8] = 1.0

# ---- proxy scenario families ----
def scen(S, G):
    zm = S.mean(2); srt = np.sort(zm,1)[:,::-1]
    em = np.zeros((4,4),bool); em[0]=em[3]=True; em[:,0]=em[:,3]=True
    gm = G.mean(3)
    rr=np.arange(4)[None,:,None,None]; cc=np.arange(4)[None,None,:,None]
    w=G+1e-6; tot=w.sum((1,2)); cr=(w*rr).sum((1,2))/tot; cc_=(w*cc).sum((1,2))/tot
    t=np.arange(30)-14.5
    vr=(cr*t).sum(1)/(t**2).sum(); vc=(cc_*t).sum(1)/(t**2).sum()
    return np.stack([srt[:,0]/(zm.mean(1)+1e-6), srt[:,1]/(srt[:,0]+1e-6), S.mean((1,2)),
        S[:,:,-10:].mean((1,2))-S[:,:,:10].mean((1,2)),
        S[:,:,-10:].mean((1,2))/(S[:,:,:10].mean((1,2))+1e-6),
        gm[:,em].mean(1)/(gm.mean((1,2))+1e-6), np.hypot(vr,vc),
        np.hypot(np.diff(cr,axis=1),np.diff(cc_,axis=1)).mean(1)],1)
F = scen(S,G); Fz=(F-F.mean(0))/(F.std(0)+1e-8)
fam = KMeans(5, n_init=10, random_state=0).fit_predict(Fz)
print("proxy family sizes", np.bincount(fam))

def full(oz, om, lam, T=1.0, Tm=1.0):
    p = oz ** (1.0/T); p = p/p.sum(2, keepdims=True)
    q = om ** (1.0/Tm); q = q/q.sum(2, keepdims=True)
    cells, msel, dsel = decode(p, q, od, lam=lam, force_dc=True)
    al = np.take_along_axis(cell2alias, cells, 1)
    rows = score_arrays(al, dsel, msel, Z, D, M, RC, reduce=False)
    fm = np.array([rows[fam==c].mean() for c in range(5)])
    b20 = np.sort(rows)[:int(0.2*N)].mean()
    return 0.70*rows.mean()+0.18*fm.min()+0.12*b20, rows.mean(), fm, b20

best=None
for lam in [0.0,0.5,1.0,1.5,2.0,3.0]:
    for T in [0.7,0.85,1.0,1.2,1.5]:
        f,o,fm,b = full(oz,om,lam,T)
        if best is None or f>best[0]: best=(f,lam,T,o,fm,b)
print(f"BEST family-aware final {best[0]:.4f} (lam {best[1]} T {best[2]}) overall {best[3]:.4f} b20 {best[5]:.4f}")
print("  family means", np.round(best[4],4), "worst", round(best[4].min(),4))
f,o,fm,b = full(oz,om,1.0,1.0)
print(f"ref lam1 T1: final {f:.4f} overall {o:.4f} worst {fm.min():.4f} b20 {b:.4f}")
for Tm in [0.8,1.0,1.3]:
    f,o,fm,b = full(oz,om,best[1],best[2],Tm); print(f"  tier T {Tm}: final {f:.4f}")
# per-family detail
cells,msel,dsel = decode(oz,om,od,lam=best[1],force_dc=True)
al=np.take_along_axis(cell2alias,cells,1)
rows=score_arrays(al,dsel,msel,Z,D,M,RC,reduce=False)
names=["peakiness","second","lvl","trend","quiet","edge","drift","wander"]
print("\nfamily profile (z-scored means) and score:")
for c in range(5):
    s=fam==c
    print(f" c{c} n={s.sum():4d} score {rows[s].mean():.4f} zoneH0 {(cells[s,0]==ZC[s,0]).mean():.3f} | " +
          " ".join(f"{n}:{Fz[s,i].mean():+.2f}" for i,n in enumerate(names)))
for nm,i in [("lvl",2),("peakiness",0),("second",1),("quiet",4)]:
    v=F[:,i]; q=np.quantile(v,[.2,.4,.6,.8]); b=np.digitize(v,q)
    print(f" {nm:10s} " + " ".join(f"q{j}:{rows[b==j].mean():.3f}" for j in range(5)))
