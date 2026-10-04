import numpy as np
from common import to_grid
from feats import CR, A1, A2, NNB, zone_features as zf_base, row_features

def zone_features2(S, RC):
    """base features + raw recent series, EWMAs, periodicity, and rank dynamics."""
    N = S.shape[0]
    X0, n0 = zf_base(S, RC)
    G = to_grid(S, RC).reshape(N, 16, 30)
    gmean = G.mean((1, 2))[:, None] + 1e-6
    f, names = [], []
    def add(a, n): f.append(np.asarray(a, np.float32)); names.append(n)

    # raw recent values (normalised) - lets the model see the actual shape
    for j in range(1, 13):
        add(G[:, :, -j] / gmean, f"raw_m{j}")
    # EWMA at several decay rates
    for tau in [1.0, 2.0, 4.0, 8.0]:
        w = np.exp(-np.arange(29, -1, -1) / tau); w /= w.sum()
        e = (G * w[None, None, :]).sum(2)
        add(e / gmean, f"ewma{tau}")
        r = (-e).argsort(1).argsort(1); add(r, f"ewma{tau}_rank")
    # rank dynamics: how the cell's rank moved between windows
    def rk(a): return (-a).argsort(1).argsort(1).astype(np.float32)
    r3 = rk(G[:, :, -3:].mean(2)); r8 = rk(G[:, :, -11:-3].mean(2)); r15 = rk(G[:, :, -30:-11].mean(2))
    add(r3 - r8, "rank_delta_3_8"); add(r8 - r15, "rank_delta_8_15")
    add((r3 == 0).astype(np.float32), "is_top3w"); add((r8 == 0).astype(np.float32), "is_top8w")
    # how long has this cell been the leader
    lead = (G == G.max(1, keepdims=True)).astype(np.float32)
    add(lead[:, :, -5:].mean(2), "lead_frac5"); add(lead[:, :, -15:].mean(2), "lead_frac15")
    add(lead.mean(2), "lead_frac30")
    # sustained activity
    thr = np.percentile(G, 75, axis=(1, 2))[:, None, None]
    add((G[:, :, -10:] > thr).mean(2), "hot_frac10"); add((G > thr).mean(2), "hot_frac30")
    # volatility of the cell relative to row
    add(G[:, :, -10:].std(2) / (G[:, :, -10:].mean(2) + 1e-6), "cv10")
    # peak recency weighted
    idx = np.arange(30, dtype=np.float32)
    wsum = (G * idx[None, None, :]).sum(2) / (G.sum(2) + 1e-6)
    add(wsum, "time_centroid")
    # cross-correlation with the global trace (does this cell drive the row?)
    tot = G.sum(1)
    tz = (tot - tot.mean(1, keepdims=True)) / (tot.std(1, keepdims=True) + 1e-6)
    gz = (G - G.mean(2, keepdims=True)) / (G.std(2, keepdims=True) + 1e-6)
    add((gz * tz[:, None, :]).mean(2), "corr_with_trace")
    X1 = np.stack(f, -1)
    return np.concatenate([X0, X1], -1), n0 + names
