"""Canonical source-proxy group folds. Written once, used by every model."""
import numpy as np
from common import load
from sklearn.cluster import KMeans

def sig(S):
    f = [np.sort(S.mean(2), 1), S.sum(1),
         np.stack([S.mean((1,2)), S.std((1,2)), S.max((1,2)),
                   np.diff(S, axis=2).std((1,2)), S.mean(2).std(1)], 1)]
    return np.concatenate([(x - x.mean(0)) / (x.std(0) + 1e-8) for x in f], 1)

def make(S, folds=5, k=14, seed=0):
    grp = KMeans(k, n_init=8, random_state=seed).fit_predict(sig(S))
    loadv = np.zeros(folds); fo = {}
    for g in np.argsort(-np.bincount(grp)):
        q = loadv.argmin(); fo[g] = q; loadv[q] += (grp == g).sum()
    return np.array([fo[g] for g in grp]), grp

if __name__ == "__main__":
    d = load("../HEN_DATA/train.csv")
    fold, grp = make(d["strip"])
    np.save("fold_group.npy", fold); np.save("grp.npy", grp)
    print("fold sizes", np.bincount(fold), "n groups", grp.max() + 1)
