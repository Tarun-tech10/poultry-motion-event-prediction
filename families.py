import numpy as np, sys
from common import *
from scorer import score_arrays
from decode import decode
from sklearn.cluster import KMeans

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC)
cell_of = (RC[:,:,0].astype(np.int64)*4 + RC[:,:,1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
cell2alias = np.zeros((N,16), np.int64)
np.put_along_axis(cell2alias, cell_of, np.arange(16)[None,:].repeat(N,0), 1)

def scenario_feats(S, G):
    zm = S.mean(2)                                   # N,16
    srt = np.sort(zm,1)[:, ::-1]
    peakiness = srt[:,0]/(zm.mean(1)+1e-6)
    second = srt[:,1]/(srt[:,0]+1e-6)                # multi_peak indicator
    lvl = S.mean((1,2))
    trend = S[:,:,-10:].mean((1,2)) - S[:,:,:10].mean((1,2))
    quiet = S[:,:,-10:].mean((1,2))/(S[:,:,:10].mean((1,2))+1e-6)
    edge = np.zeros(len(S))
    em = np.zeros((4,4),bool); em[0]=em[3]=True; em[:,0]=em[:,3]=True
    gm = G.mean(3)
    edge = gm[:,em].mean(1)/(gm.mean((1,2))+1e-6)
    # centroid drift magnitude
    rr=np.arange(4)[None,:,None,None]; cc=np.arange(4)[None,None,:,None]
    w=G+1e-6; tot=w.sum((1,2))
    cr=(w*rr).sum((1,2))/tot; cc_=(w*cc).sum((1,2))/tot
    t=np.arange(30)-14.5
    vr=(cr*t).sum(1)/(t**2).sum(); vc=(cc_*t).sum(1)/(t**2).sum()
    drift=np.hypot(vr,vc)
    wander = np.hypot(np.diff(cr,axis=1),np.diff(cc_,axis=1)).mean(1)
    return np.stack([peakiness, second, lvl, trend, quiet, edge, drift, wander],1), \
           ["peakiness","second","lvl","trend","quiet","edge","drift","wander"]

F, names = scenario_feats(S, G)
Fz = (F-F.mean(0))/(F.std(0)+1e-8)
km = KMeans(5, n_init=10, random_state=0).fit(Fz)
fam = km.labels_
print("proxy family sizes", np.bincount(fam))
print("centroids (z):")
print("      "+" ".join(f"{n:>9}" for n in names))
for c in range(5):
    print(f" c{c}  "+" ".join(f"{v:9.2f}" for v in km.cluster_centers_[c]))
np.save("fam.npy", fam)

seed = sys.argv[1] if len(sys.argv)>1 else "0"
try:
    oz=np.load(f"oof_z_{seed}.npy"); om=np.load(f"oof_m_{seed}.npy"); od=np.load(f"oof_d_{seed}.npy")
except FileNotFoundError:
    print("no OOF yet"); sys.exit()
cells, msel, dsel = decode(oz, om, od, lam=float(sys.argv[2]) if len(sys.argv)>2 else 0.0, force_dc=True)
al = np.take_along_axis(cell2alias, cells, 1)
rows = score_arrays(al, dsel, msel, Z, D, M, RC, reduce=False)
print(f"\nOVERALL {rows.mean():.4f}  bottom20 {np.sort(rows)[:int(0.2*N)].mean():.4f}")
print("by proxy family:")
for c in range(5):
    s=fam==c; print(f"  c{c} n={s.sum():4d} score {rows[s].mean():.4f}  zoneacc H0 {(cells[s,0]==ZC[s,0]).mean():.3f}")
print("worst proxy family:", min(rows[fam==c].mean() for c in range(5)).round(4))
for nm, v in zip(names, F.T):
    q = np.quantile(v, [0,.2,.4,.6,.8,1.]); b=np.clip(np.digitize(v,q[1:-1]),0,4)
    print(f"  {nm:10s} " + " ".join(f"q{i}:{rows[b==i].mean():.3f}" for i in range(5)))
