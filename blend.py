import numpy as np
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

ez = np.load("ens_oof_z.npy"); om = np.load("ens_oof_m.npy")
rk = np.load("zone_rank.npy")            # raw lambdarank scores
od = np.zeros((N,3,9), np.float32); od[:,:,8]=1.0

def softmax(x, T):
    e = np.exp((x - x.max(-1, keepdims=True)) / T)
    return e / e.sum(-1, keepdims=True)

# pick rank temperature by OOF log-loss
best_T = None
for T in [0.3,0.5,0.75,1.0,1.5,2.0,3.0]:
    p = softmax(rk, T)
    ll = -np.mean([np.log(p[np.arange(N), k, ZC[:,k]] + 1e-9).mean() for k in range(3)])
    if best_T is None or ll < best_T[1]: best_T = (T, ll)
print("rank softmax T", best_T)
pr = softmax(rk, best_T[0])

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
fam = KMeans(5, n_init=10, random_state=0).fit_predict(Fz)

def ev(pz, lam=1.5, T=0.9, Tm=1.15, beta=0.002, beta_m=0.02):
    p = pz**(1.0/T); p/=p.sum(2,keepdims=True)
    q = om**(1.0/Tm); q/=q.sum(2,keepdims=True)
    cells,msel,dsel = decode(p,q,od,lam=lam,force_dc=True,beta=beta,beta_m=beta_m)
    al = np.take_along_axis(cell2alias, cells, 1)
    rows = score_arrays(al,dsel,msel,Z,D,M,RC,reduce=False)
    fm = np.array([rows[fam==c].mean() for c in range(5)])
    b20 = np.sort(rows)[:int(0.2*N)].mean()
    return 0.70*rows.mean()+0.18*fm.min()+0.12*b20, rows.mean(), fm.min()

print("binary-only     ", np.round(ev(ez),5), "acc", [round(float((ez[:,k].argmax(1)==ZC[:,k]).mean()),4) for k in range(3)])
print("rank-only       ", np.round(ev(pr),5))
for w in [0.15,0.25,0.35,0.5]:
    lo = (1-w)*np.log(np.clip(ez,1e-7,1)) + w*np.log(np.clip(pr,1e-7,1))
    e = np.exp(lo - lo.max(2,keepdims=True)); bz = (e/e.sum(2,keepdims=True)).astype(np.float32)
    print(f"blend w={w}    ", np.round(ev(bz),5), "acc", [round(float((bz[:,k].argmax(1)==ZC[:,k]).mean()),4) for k in range(3)])
