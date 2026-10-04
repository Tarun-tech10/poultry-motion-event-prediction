import numpy as np, collections
from common import *

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); ar = np.arange(N)[:,None]
pr = RC[ar, Z, 0].astype(int); pc = RC[ar, Z, 1].astype(int)   # N,3 grid pos of target zone

print("=== P(direction | grid_row of target zone), horizon-pooled ===")
print("row  " + " ".join(f"{x:>6}" for x in ALLD) + "   n")
for r in range(4):
    sel = (pr==r)
    cnt = np.array([ (D[sel]==k).sum() for k in range(9)], float)
    print(f" {r}   " + " ".join(f"{v:6.3f}" for v in cnt/cnt.sum()) + f"  {int(cnt.sum())}")
print("\ncol  " + " ".join(f"{x:>6}" for x in ALLD))
for c in range(4):
    sel = (pc==c)
    cnt = np.array([ (D[sel]==k).sum() for k in range(9)], float)
    print(f" {c}   " + " ".join(f"{v:6.3f}" for v in cnt/cnt.sum()))

# net inward tendency: for zones at row0 (top edge) is DS-ish more common than DN-ish?
NORTH={0,1,7}; SOUTH={3,4,5}; EAST={1,2,3}; WEST={5,6,7}
for r in range(4):
    sel=(pr==r); n=(np.isin(D[sel],list(NORTH))).mean(); s=(np.isin(D[sel],list(SOUTH))).mean()
    print(f"row{r}: P(north-ish) {n:.3f}  P(south-ish) {s:.3f}")
for c in range(4):
    sel=(pc==c); e=(np.isin(D[sel],list(EAST))).mean(); w=(np.isin(D[sel],list(WEST))).mean()
    print(f"col{c}: P(east-ish) {e:.3f}  P(west-ish) {w:.3f}")

print("\n=== P(DC | position) and P(DC | tier) ===")
for r in range(4):
    print(f" row{r} P(DC)={ (D[pr==r]==8).mean():.3f}", end="")
print()
for t in range(4):
    print(f" M{t} P(DC)={ (D[M==t]==8).mean():.3f} n={(M==t).sum()}", end="")
print()
# edge vs interior
edge = (pr==0)|(pr==3)|(pc==0)|(pc==3)
print("P(DC|edge)",(D[edge]==8).mean().round(3),"P(DC|interior)",(D[~edge]==8).mean().round(3))

print("\n=== direction vs displacement to NEXT-horizon zone, conditioned on being a mover ===")
for k in range(2):
    dr=pr[:,k+1]-pr[:,k]; dc=pc[:,k+1]-pc[:,k]
    mv=(dr!=0)|(dc!=0)
    # angle of actual displacement
    ang=np.arctan2(dc[mv],-dr[mv]); tk=(np.round(ang/(np.pi/4)).astype(int))%8
    dd=D[mv,k]
    ok=(dd<8)
    cdist=np.minimum((tk[ok]-dd[ok])%8,(dd[ok]-tk[ok])%8)
    print(f" H{k}: compass dist(D, actual displacement):", collections.Counter(cdist.tolist()).most_common())

print("\n=== tier vs zone strength relative: try to find the quantiser ===")
zs = S[ar, Z, :]
print(" tier vs target-zone last-6 mean:")
for t in range(4):
    s=(M[:,0]==t); print(f"  M{t}: n={s.sum():5d} zone_last6 {zs[s,0,-6:].mean():6.2f}  zone_all {zs[s,0].mean():6.2f} global {S[s].mean():5.2f}")
# is tier maybe a global (row-level) property rather than zone-level?
print(" tier agreement across horizons: M0==M1", (M[:,0]==M[:,1]).mean().round(3), "M0==M2",(M[:,0]==M[:,2]).mean().round(3))
