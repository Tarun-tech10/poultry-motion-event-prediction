import numpy as np, collections
from common import *

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC)          # N,4,4,30
rr = np.arange(4)[None,:,None,None]; cc = np.arange(4)[None,None,:,None]

def centroid(g):  # g: N,4,4,T -> N,T,2
    w = g + 1e-6
    tot = w.sum((1,2))
    return np.stack([(w*rr).sum((1,2))/tot, (w*cc).sum((1,2))/tot], -1)

C = centroid(G)   # N,30,2

def dir_of(vec, thr):
    """vec: N,2 (drow,dcol) -> index into ALLD"""
    mag = np.hypot(vec[:,0], vec[:,1])
    ang = np.arctan2(vec[:,1], -vec[:,0])          # 0 = north(-row), pi/2 = east(+col)
    k = np.round(ang/(np.pi/4)).astype(int) % 8    # 0=DN,1=DNE,...
    k = np.where(mag < thr, 8, k)
    return k

print("=== does PAST centroid drift predict D[H0]? ===")
for (a,b,lbl) in [(0,30,"full"),(15,30,"last15"),(20,30,"last10"),(24,30,"last6")]:
    t = np.arange(b-a); t = (t-t.mean())
    seg = C[:,a:b,:]
    slope = (seg*t[None,:,None]).sum(1)/ (t**2).sum()   # per-frame velocity
    for thr in [0.0, 0.01, 0.03]:
        pk = dir_of(slope*6, thr)
        acc = (pk==D[:,0]).mean()
        # adjacent credit
        adj = ((pk<8)&(D[:,0]<8)&(np.minimum((pk-D[:,0])%8,(D[:,0]-pk)%8)==1)).mean()
        print(f"  {lbl} thr={thr}: exact {acc:.4f} adj {adj:.4f}")

print("\n=== confusion between past-drift dir and D[H0] (last10, thr .01) ===")
t=np.arange(10); t=t-t.mean(); slope=(C[:,20:30,:]*t[None,:,None]).sum(1)/(t**2).sum()
pk = dir_of(slope*6, 0.01)
cm = np.zeros((9,9),int)
for a,b in zip(pk, D[:,0]): cm[a,b]+=1
print("      "+" ".join(f"{x:>5}" for x in ALLD))
for i in range(9): print(f"{ALLD[i]:>5} "+" ".join(f"{cm[i,j]:5d}" for j in range(9)))

print("\n=== is D related to intra-horizon PEAK-zone shift? check past analogue autocorr ===")
# build pseudo-horizons from past: [12:18],[18:24],[24:30]
def peak_shift_dir(g, a, b):
    h = g[:,:,:,a:b]
    n = h.shape[3]//2
    f1 = h[:,:,:,:n].mean(3).reshape(-1,16); f2 = h[:,:,:,n:].mean(3).reshape(-1,16)
    i1 = f1.argmax(1); i2 = f2.argmax(1)
    return np.stack([i2//4 - i1//4, i2%4 - i1%4],-1).astype(np.float32)
for a,b in [(24,30),(18,24)]:
    v = peak_shift_dir(G,a,b)
    for thr in [0.0,0.5]:
        pk = dir_of(v,thr); print(f"  peakshift[{a}:{b}] thr{thr}: vs D[H0] exact {(pk==D[:,0]).mean():.4f}  DCfrac {(pk==8).mean():.3f}")

print("\n=== direction autocorrelation within ledger ===")
print("P(D0==D1)", (D[:,0]==D[:,1]).mean().round(4), " P(D1==D2)", (D[:,1]==D[:,2]).mean().round(4))
nz = (D[:,0]<8)&(D[:,1]<8)
print("among non-DC pairs P(D0==D1)", (D[nz,0]==D[nz,1]).mean().round(4), "n=",nz.sum())
cd = np.minimum((D[nz,0]-D[nz,1])%8,(D[nz,1]-D[nz,0])%8)
print("compass dist D0 vs D1 dist:", collections.Counter(cd.tolist()).most_common())

print("\n=== tier definition probe ===")
zs_peak = S[np.arange(N)[:,None], Z, :].max(axis=2)
zs_mean = S[np.arange(N)[:,None], Z, :].mean(axis=2)
glob = S.mean((1,2))
for t in range(4):
    s = M[:,0]==t
    print(f" M{t}: zonepast_mean {zs_mean[s,0].mean():6.2f} zonepast_peak {zs_peak[s,0].mean():6.2f} glob {glob[s].mean():5.2f} ratio {(zs_mean[s,0]/glob[s]).mean():.3f}")
