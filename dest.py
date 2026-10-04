"""Where do movers GO?  Look for a systematic destination bias exploitable without per-row signal."""
import numpy as np, collections
from common import *
from evalbal import rule_families

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC).reshape(N, 16, 30)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
CRc = np.array([[i // 4, i % 4] for i in range(16)])
fam = rule_families(ZC, M)
mvr = fam == 0
m3 = G[:, :, -3:].mean(2)
order = np.argsort(-m3, 1)                      # past ranking of cells
rank_of = np.argsort(order, 1)                  # rank_of[i,cell]
hot = order[:, 0]

def prof(sel, lbl):
    print(f"\n=== {lbl} (n={sel.sum()}) ===")
    for k in range(3):
        r = rank_of[np.arange(N), ZC[:, k]][sel]
        cnt = collections.Counter(r.tolist())
        tot = len(r)
        print(f"  H{k} past-rank of true zone: " + " ".join(f"r{i}:{cnt.get(i,0)/tot:.3f}" for i in range(6)))
    # centrality of destination
    pos = CRc[ZC[:, 2]][sel]
    cen = (np.abs(pos - 1.5).sum(1) <= 1.0).mean()          # inner 2x2
    hp = CRc[hot][sel]; hcen = (np.abs(hp - 1.5).sum(1) <= 1.0).mean()
    print(f"  H2 dest in inner 2x2: {cen:.3f}   (hot cell in inner 2x2: {hcen:.3f}, uniform 0.25)")
    edge = ((pos[:, 0] == 0) | (pos[:, 0] == 3) | (pos[:, 1] == 0) | (pos[:, 1] == 3)).mean()
    print(f"  H2 dest on edge ring: {edge:.3f}  (uniform 0.75)")
    # direction of displacement relative to grid centre (does it move inward or outward?)
    hv = CRc[hot][sel] - 1.5; dv = pos - CRc[hot][sel]
    rad = (hv * dv).sum(1) / (np.linalg.norm(hv, axis=1) + 1e-6)
    print(f"  radial component of move (+ = outward, - = toward centre): {rad.mean():+.4f}")
    # is the destination the 2nd/3rd hottest?
    print(f"  H2 dest == 2nd hottest: {(ZC[sel,2]==order[sel,1]).mean():.3f}   == 3rd: {(ZC[sel,2]==order[sel,2]).mean():.3f}")
    # opposite/mirror hypotheses
    mir_r = (3 - CRc[hot][sel][:, 0]) * 4 + CRc[hot][sel][:, 1]
    mir_c = CRc[hot][sel][:, 0] * 4 + (3 - CRc[hot][sel][:, 1])
    mir_b = (3 - CRc[hot][sel][:, 0]) * 4 + (3 - CRc[hot][sel][:, 1])
    print(f"  H2 dest == row-mirror {(ZC[sel,2]==mir_r).mean():.3f}  col-mirror {(ZC[sel,2]==mir_c).mean():.3f}  point-mirror {(ZC[sel,2]==mir_b).mean():.3f} (chance .0625)")

prof(np.ones(N, bool), "ALL ROWS")
prof(mvr, "crowd_shift (movers)")

# marginal destination distribution over the 16 grid cells, movers only
print("\nH2 destination distribution over grid cells (movers), counts:")
cnt = np.bincount(ZC[mvr, 2], minlength=16).reshape(4, 4)
print(cnt, " chance per cell:", round(mvr.sum() / 16, 1))
print("\nH2 destination by (grid_row,grid_col) marginals, movers:")
print("  row:", np.round(np.bincount(CRc[ZC[mvr,2]][:,0], minlength=4) / mvr.sum(), 3),
      " col:", np.round(np.bincount(CRc[ZC[mvr,2]][:,1], minlength=4) / mvr.sum(), 3))
print("  (uniform = 0.25 each; grid transform is random per row so any skew is a generator bias)")
