import numpy as np, itertools, json
from common import *
from scorer import score_arrays
from decode import decode
from sklearn.cluster import KMeans

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC)
cell_of = (RC[:,:,0].astype(np.int64)*4 + RC[:,:,1].astype(np.int64))
cell2alias = np.zeros((N,16), np.int64)
np.put_along_axis(cell2alias, cell_of, np.arange(16)[None,:].repeat(N,0), 1)
oz = np.load("ens_oof_z.npy"); om = np.load("ens_oof_m.npy")
od = np.zeros((N,3,9), np.float32); od[:,:,8] = 1.0

def scen(S,G):
    zm=S.mean(2); srt=np.sort(zm,1)[:,::-1]
    em=np.zeros((4,4),bool); em[0]=em[3]=True; em[:,0]=em[:,3]=True
    gm=G.mean(3); rr=np.arange(4)[None,:,None,None]; cc=np.arange(4)[None,None,:,None]
    w=G+1e-6; tot=w.sum((1,2)); cr=(w*rr).sum((1,2))/tot; cc_=(w*cc).sum((1,2))/tot
    t=np.arange(30)-14.5; vr=(cr*t).sum(1)/(t**2).sum(); vc=(cc_*t).sum(1)/(t**2).sum()
    return np.stack([srt[:,0]/(zm.mean(1)+1e-6), srt[:,1]/(srt[:,0]+1e-6), S.mean((1,2)),
        S[:,:,-10:].mean((1,2))-S[:,:,:10].mean((1,2)),
        S[:,:,-10:].mean((1,2))/(S[:,:,:10].mean((1,2))+1e-6),
        gm[:,em].mean(1)/(gm.mean((1,2))+1e-6), np.hypot(vr,vc),
        np.hypot(np.diff(cr,axis=1),np.diff(cc_,axis=1)).mean(1)],1)
Fz = scen(S,G); Fz=(Fz-Fz.mean(0))/(Fz.std(0)+1e-8)
fams = {k: KMeans(k, n_init=10, random_state=0).fit_predict(Fz) for k in (5,)}
fam = fams[5]

def ev(lam,T,Tm,beta,beta_m):
    p = oz**(1.0/T); p/=p.sum(2,keepdims=True)
    q = om**(1.0/Tm); q/=q.sum(2,keepdims=True)
    cells,msel,dsel = decode(p,q,od,lam=lam,force_dc=True,beta=beta,beta_m=beta_m)
    al = np.take_along_axis(cell2alias, cells, 1)
    rows = score_arrays(al,dsel,msel,Z,D,M,RC,reduce=False)
    fm = np.array([rows[fam==c].mean() for c in range(5)])
    b20 = np.sort(rows)[:int(0.2*N)].mean()
    return 0.70*rows.mean()+0.18*fm.min()+0.12*b20, rows.mean(), fm.min(), b20

grid = list(itertools.product([0.5,1.0,1.5,2.0],[0.8,0.9,1.0,1.1],[1.0,1.15],
                             [0.0,0.002,0.005,0.01,0.02,0.04],[0.0,0.02,0.05,0.1]))
best=None
for g in grid:
    r=ev(*g)
    if best is None or r[0]>best[0]: best=(r[0],g,r[1],r[2],r[3])
print(f"BEST {best[0]:.4f} params lam={best[1][0]} T={best[1][1]} Tm={best[1][2]} beta={best[1][3]} beta_m={best[1][4]}")
print(f"      overall {best[2]:.4f} worst {best[3]:.4f} b20 {best[4]:.4f}")
b=ev(1.5,0.9,1.15,0.0,0.0); print(f"no-risk ref: {b[0]:.4f} overall {b[1]:.4f} worst {b[2]:.4f} b20 {b[3]:.4f}")
json.dump(dict(lam=best[1][0],T=best[1][1],Tm=best[1][2],beta=best[1][3],beta_m=best[1][4]), open("decode_params.json","w"))
print("saved decode_params.json")
