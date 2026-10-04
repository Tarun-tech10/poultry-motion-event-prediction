import numpy as np, collections
from common import *
from sklearn.cluster import KMeans

tr = load("../HEN_DATA/train.csv"); te = load("../HEN_DATA/test.csv", False)
S = np.concatenate([tr["strip"], te["strip"]], 0)
istest = np.r_[np.zeros(len(tr["strip"]),bool), np.ones(len(te["strip"]),bool)]
N = len(S); ntr=(~istest).sum(); nte=istest.sum()
print("N",N,"train",ntr,"test",nte, "chance same-split for test row:", round((nte-1)/(N-1),4))

# --- signature candidates (all invariant to zone aliasing & spatial transform)
zone_mean = np.sort(S.mean(2), 1)                 # 16 sorted zone means
zone_std  = np.sort(S.std(2), 1)
trace     = S.sum(1)                              # 30
qhist     = np.stack([np.bincount(s.astype(int).ravel(), minlength=64) for s in S]).astype(np.float32)
feats = {
  "zone_mean_sorted": zone_mean,
  "trace": trace,
  "qhist": qhist/qhist.sum(1,keepdims=True),
  "global_stats": np.stack([S.mean((1,2)), S.std((1,2)), S.max((1,2)), np.percentile(S,90,axis=(1,2)),
                            np.diff(S,axis=2).std((1,2)), S.mean(2).std(1)],1),
}
def nnsame(X, name):
    X = (X - X.mean(0)) / (X.std(0)+1e-8)
    X = X / (np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    sim = X @ X.T; np.fill_diagonal(sim,-9)
    nn = sim.argmax(1)
    for lbl, sel in [("test",istest),("train",~istest)]:
        frac = (istest[nn[sel]]==istest[sel]).mean()
        print(f"  {name:20s} NN-same-split[{lbl}] {frac:.4f}")
for k,v in feats.items(): nnsame(v,k)
allf = np.concatenate([ (f-f.mean(0))/(f.std(0)+1e-8) for f in feats.values()],1)
nnsame(allf,"ALL")

print("\n=== KMeans clusters: train/test purity ===")
X = (allf - allf.mean(0))/(allf.std(0)+1e-8)
for k in [8, 14, 19, 25]:
    km = KMeans(k, n_init=6, random_state=0).fit(X)
    lab = km.labels_
    rows=[]
    for c in range(k):
        s = lab==c; rows.append((s.sum(), istest[s].mean()))
    pur = np.mean([max(p,1-p) for _,p in rows])
    print(f" k={k}: mean purity {pur:.3f}  testfrac per cluster {sorted(round(p,2) for _,p in rows)}")
    np.save(f"km{k}.npy", lab)
