import numpy as np
from common import to_grid

CR = np.array([[r, c] for r in range(4) for c in range(4)], np.float32)   # cell -> (row,col)
MAN = np.abs(CR[:, None] - CR[None, :]).sum(-1)
A1 = (MAN == 1).astype(np.float32)
A2 = (MAN == 2).astype(np.float32)
NNB = A1.sum(1)                                                          # neighbour count 2/3/4

def zone_features(S, RC):
    """S:(N,16,30) row-local strips, RC:(N,16,2) grid coords.
    Returns X:(N,16,F) features in CELL order (cell = r*4+c), and names."""
    N = S.shape[0]
    G = to_grid(S, RC).reshape(N, 16, 30)          # now indexed by CELL
    f, names = [], []
    def add(a, n):
        f.append(a.astype(np.float32)); names.append(n)

    tot = G.sum(1) + 1e-6                          # (N,30) global per-frame
    gmean = G.mean((1, 2), keepdims=True) + 1e-6   # (N,1,1)

    wins = [1, 2, 3, 5, 8, 12, 20, 30]
    W = {}
    for w in wins:
        m = G[:, :, -w:].mean(2)                   # (N,16)
        W[w] = m
        add(m / gmean[:, :, 0], f"mean{w}_rel")
        add(m - G[:, :, -w:].mean((1, 2))[:, None], f"mean{w}_dev")
        r = (-m).argsort(1).argsort(1)             # rank 0 = strongest
        add(r, f"rank{w}")
        add(m / (m.max(1, keepdims=True) + 1e-6), f"mean{w}_share")
    # share of total motion
    for w in [3, 8, 30]:
        add(G[:, :, -w:].sum(2) / (tot[:, -w:].sum(1)[:, None] + 1e-6), f"frac{w}")
    # trends
    add((W[3] - G[:, :, -6:-3].mean(2)) / gmean[:, :, 0], "trend3")
    add((W[5] - G[:, :, -10:-5].mean(2)) / gmean[:, :, 0], "trend5")
    add((W[8] - G[:, :, -16:-8].mean(2)) / gmean[:, :, 0], "trend8")
    add((W[12] - G[:, :, :12].mean(2)) / gmean[:, :, 0], "trend_long")
    t = np.arange(30, dtype=np.float32) - 14.5
    slope = (G * t).sum(2) / (t ** 2).sum()
    add(slope / gmean[:, :, 0], "slope30")
    t10 = np.arange(10, dtype=np.float32) - 4.5
    add((G[:, :, -10:] * t10).sum(2) / (t10 ** 2).sum() / gmean[:, :, 0], "slope10")
    # shape stats
    add(G.max(2) / gmean[:, :, 0], "max_rel")
    add(G.std(2) / gmean[:, :, 0], "std_rel")
    add(G.argmax(2).astype(np.float32), "argmax_t")
    add((G[:, :, -10:] > G.mean(2, keepdims=True)).mean(2), "frac_above_own")
    add((G[:, :, -10:] == 0).mean(2), "frac_zero10")
    # autocorrelation / smoothness
    d1 = np.diff(G, axis=2)
    add(np.abs(d1[:, :, -10:]).mean(2) / gmean[:, :, 0], "absdiff10")
    add(d1[:, :, -10:].mean(2) / gmean[:, :, 0], "meandiff10")
    # ---- neighbourhood ----
    for w in [3, 8]:
        m = W[w]
        n1 = m @ A1.T / NNB[None, :]
        n2 = m @ A2.T / (A2.sum(1)[None, :] + 1e-6)
        add(n1 / gmean[:, :, 0], f"nb1_{w}")
        add(n2 / gmean[:, :, 0], f"nb2_{w}")
        add(m / (n1 + 1e-6), f"self_over_nb1_{w}")
        add((m @ A1.T).max(1, keepdims=True).repeat(16, 1) * 0 + (m[:, None, :] * A1[None]).max(2) / gmean[:, :, 0], f"nbmax_{w}")
    # neighbour trend (is motion flowing in?)
    tr3 = (W[3] - G[:, :, -6:-3].mean(2))
    add((tr3 @ A1.T) / NNB[None, :] / gmean[:, :, 0], "nb_trend3")
    # ---- geometry (transform-invariant only) ----
    add(np.broadcast_to(NNB[None, :], (N, 16)), "n_neighbors")
    dc = np.abs(CR - 1.5).sum(1)
    add(np.broadcast_to(dc[None, :], (N, 16)), "dist_center")
    add(np.broadcast_to((NNB == 2).astype(np.float32)[None, :], (N, 16)), "is_corner")
    # ---- centroid drift: is the crowd heading here? ----
    wts = G + 1e-6
    cen = np.einsum("nct,cd->ntd", wts, CR) / wts.sum(1)[:, :, None]      # (N,30,2)
    tt = np.arange(30, dtype=np.float32) - 14.5
    vel = (cen * tt[None, :, None]).sum(1) / (tt ** 2).sum()              # (N,2)
    t10b = np.arange(10, dtype=np.float32) - 4.5
    vel10 = (cen[:, -10:] * t10b[None, :, None]).sum(1) / (t10b ** 2).sum()
    last = cen[:, -3:].mean(1)                                            # (N,2)
    rel = CR[None, :, :] - last[:, None, :]                               # (N,16,2)
    dist = np.linalg.norm(rel, axis=2) + 1e-6
    add(dist, "dist_to_centroid")
    for v, nm in [(vel, "vel30"), (vel10, "vel10")]:
        proj = (rel * v[:, None, :]).sum(2) / dist
        add(proj, f"proj_{nm}")
        add(np.broadcast_to(np.linalg.norm(v, axis=1)[:, None], (N, 16)), f"speed_{nm}")
        # predicted centroid 6/12/18 frames ahead -> distance to this cell
        for hstep, hn in [(6, "h0"), (12, "h1"), (18, "h2")]:
            pred = last + v * hstep
            add(np.linalg.norm(CR[None] - pred[:, None, :], axis=2), f"dist_pred_{nm}_{hn}")
    # ---- row-level context broadcast ----
    srt = np.sort(W[3], 1)[:, ::-1]
    ctx = {
        "g_level": G.mean((1, 2)), "g_std": G.std((1, 2)),
        "g_trend": G[:, :, -10:].mean((1, 2)) - G[:, :, :10].mean((1, 2)),
        "g_peakiness": srt[:, 0] / (W[3].mean(1) + 1e-6),
        "g_second_ratio": srt[:, 1] / (srt[:, 0] + 1e-6),
        "g_third_ratio": srt[:, 2] / (srt[:, 0] + 1e-6),
        "g_entropy": -(W[3] / W[3].sum(1, keepdims=True) * np.log(W[3] / W[3].sum(1, keepdims=True) + 1e-9)).sum(1),
        "g_zerofrac": (G == 0).mean((1, 2)),
        "g_maxlevel": G.max((1, 2)),
        "g_tracestd": tot.std(1) / (tot.mean(1) + 1e-6),
    }
    for k, v in ctx.items():
        add(np.broadcast_to(v[:, None].astype(np.float32), (N, 16)), k)
    X = np.stack(f, -1)                     # (N,16,F)
    return X, names

def row_features(S, RC):
    """row-level features for the tier model"""
    N = S.shape[0]
    G = to_grid(S, RC).reshape(N, 16, 30)
    m3 = G[:, :, -3:].mean(2); srt = np.sort(m3, 1)[:, ::-1]
    tot = G.sum(1)
    out, names = [], []
    def add(a, n): out.append(np.asarray(a, np.float32)); names.append(n)
    for w in [1, 2, 3, 5, 8, 15, 30]:
        add(G[:, :, -w:].mean((1, 2)), f"lvl{w}")
        add(G[:, :, -w:].max((1, 2)), f"max{w}")
        add(np.sort(G[:, :, -w:].mean(2), 1)[:, -1], f"top1_{w}")
        add(np.sort(G[:, :, -w:].mean(2), 1)[:, -3:].mean(1), f"top3_{w}")
    add(G[:, :, -10:].mean((1, 2)) - G[:, :, :10].mean((1, 2)), "trend")
    add(G[:, :, -5:].mean((1, 2)) / (G[:, :, -20:-5].mean((1, 2)) + 1e-6), "ratio_recent")
    add(tot.std(1) / (tot.mean(1) + 1e-6), "trace_cv")
    add(np.diff(tot, axis=1).std(1) / (tot.mean(1) + 1e-6), "trace_dcv")
    add(srt[:, 0] / (m3.mean(1) + 1e-6), "peakiness")
    add(srt[:, 1] / (srt[:, 0] + 1e-6), "second_ratio")
    add((G == 0).mean((1, 2)), "zerofrac")
    add(G.std((1, 2)), "gstd")
    add(np.percentile(G, 90, axis=(1, 2)), "p90")
    add(np.percentile(G[:, :, -10:], 99, axis=(1, 2)), "p99_10")
    # trend of the strongest zones
    top = np.argsort(-m3, 1)[:, :3]
    tg = np.take_along_axis(G, top[:, :, None], 1)
    add(tg[:, :, -5:].mean((1, 2)) - tg[:, :, -15:-5].mean((1, 2)), "top_trend")
    add(tg[:, :, -3:].mean((1, 2)), "top_recent")
    return np.stack(out, 1), names
