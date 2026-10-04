import numpy as np, collections
from common import *

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z)
G = to_grid(S, RC)  # N,4,4,30
print("grid", G.shape, "mean", G.mean())

# position of each target zone in grid
pr = RC[np.arange(N)[:,None], Z, 0].astype(int)  # N,3 grid_row
pc = RC[np.arange(N)[:,None], Z, 1].astype(int)

# --- H1: does direction encode the zone->zone transition?
for k in range(2):
    dr = pr[:,k+1]-pr[:,k]; dc = pc[:,k+1]-pc[:,k]
    same = (dr==0)&(dc==0)
    print(f"\n-- H{k}->H{k+1}: same zone frac {same.mean():.3f}; DC at H{k} frac {(D[:,k]==8).mean():.3f}")
    print("   P(D==DC | same zone) =", (D[same,k]==8).mean().round(3),
          " P(same zone | D==DC) =", (same[D[:,k]==8]).mean().round(3))
    # for movers, does D match the displacement sign?
    mv = ~same
    agree=0; tot=0
    for name,(vr,vc) in DVEC.items():
        sel = mv & (D[:,k]==ALLD.index(name))
        tot += sel.sum()
        agree += ((np.sign(dr[sel])==vr)&(np.sign(dc[sel])==vc)).sum()
    print(f"   movers: dir matches sign(displacement to next zone): {agree}/{tot} = {agree/max(tot,1):.3f}")

# --- alternative: dir at Hk matches displacement from PREVIOUS zone
for k in range(1,3):
    dr = pr[:,k]-pr[:,k-1]; dc = pc[:,k]-pc[:,k-1]
    same=(dr==0)&(dc==0)
    agree=0;tot=0
    for name,(vr,vc) in DVEC.items():
        sel = (~same)&(D[:,k]==ALLD.index(name)); tot+=sel.sum()
        agree += ((np.sign(dr[sel])==vr)&(np.sign(dc[sel])==vc)).sum()
    print(f"-- backward: D[H{k}] vs prev->cur displacement: {agree}/{max(tot,1)} = {agree/max(tot,1):.3f}; P(DC|same)= {(D[same,k]==8).mean():.3f}")

# --- Tier vs motion magnitude: proxy using past overall level
lvl = G.reshape(N,16,30).mean(axis=(1,2))
for t in range(4):
    print("tier M%d at H0: mean past level %.2f  n=%d"%(t, lvl[M[:,0]==t].mean(), (M[:,0]==t).sum()))
# tier vs the target zone's own past strength
zs = S[np.arange(N)[:,None], Z, :].mean(axis=2)  # N,3 mean past motion in target zone
for t in range(4):
    print("tier M%d: target-zone past mean %.2f"%(t, zs[:,0][M[:,0]==t].mean()))

# --- zone vs past argmax
last = S[:,:,-6:].mean(axis=2)
am = last.argmax(1)
print("\nP(H0 zone == argmax last6 past) =", (am==Z[:,0]).mean().round(4))
for w in [3,6,10,30]:
    am = S[:,:,-w:].mean(axis=2).argmax(1)
    for k in range(3):
        print(f"  win{w} -> H{k} acc {(am==Z[:,k]).mean():.4f}", end="")
    print()
# manhattan dist of argmax to truth
amr = RC[np.arange(N), am, 0].astype(int); amc = RC[np.arange(N), am, 1].astype(int)
md = np.abs(amr-pr[:,0])+np.abs(amc-pc[:,0])
print("manhattan dist argmax->H0 zone:", collections.Counter(md.tolist()).most_common())
print("zone persistence: H0==H1 %.3f H1==H2 %.3f H0==H2 %.3f"%((Z[:,0]==Z[:,1]).mean(),(Z[:,1]==Z[:,2]).mean(),(Z[:,0]==Z[:,2]).mean()))
print("tier persistence H0==H1 %.3f"%((M[:,0]==M[:,1]).mean()))
