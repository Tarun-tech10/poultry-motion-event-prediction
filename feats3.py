import numpy as np
from common import to_grid
from feats import CR, A1, A2, NNB
from feats2 import zone_features2

SH = np.array([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], np.float32)

def _shift_field(F, v):
    """F:(N,4,4) advect by vector v:(N,2) using bilinear sampling (source = pos - v)."""
    N = F.shape[0]
    rr, cc = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    sr = rr[None] - v[:, 0][:, None, None]
    sc = cc[None] - v[:, 1][:, None, None]
    sr = np.clip(sr, 0, 3); sc = np.clip(sc, 0, 3)
    r0 = np.floor(sr).astype(int); c0 = np.floor(sc).astype(int)
    r1 = np.minimum(r0 + 1, 3); c1 = np.minimum(c0 + 1, 3)
    fr = sr - r0; fc = sc - c0
    idx = np.arange(N)[:, None, None]
    out = (F[idx, r0, c0] * (1 - fr) * (1 - fc) + F[idx, r1, c0] * fr * (1 - fc)
           + F[idx, r0, c1] * (1 - fr) * fc + F[idx, r1, c1] * fr * fc)
    return out

def flow_vec(G4, lag):
    """G4:(N,4,4,T) -> soft translation per frame, estimated by shift cross-correlation."""
    N, _, _, T = G4.shape
    a = G4[:, :, :, :T - lag]; b = G4[:, :, :, lag:]
    a = a - a.mean((1, 2), keepdims=True); b = b - b.mean((1, 2), keepdims=True)
    sc = np.zeros((N, 9), np.float32)
    for i, (dr, dc) in enumerate(SH):
        dr = int(dr); dc = int(dc)
        ar0, ar1 = max(0, -dr), 4 - max(0, dr)
        ac0, ac1 = max(0, -dc), 4 - max(0, dc)
        aa = a[:, ar0:ar1, ac0:ac1]
        bb = b[:, ar0 + dr:ar1 + dr, ac0 + dc:ac1 + dc]
        sc[:, i] = (aa * bb).sum((1, 2, 3)) / (aa.shape[1] * aa.shape[2])
    sc = sc - sc.min(1, keepdims=True)
    w = sc / (sc.sum(1, keepdims=True) + 1e-6)
    v = w @ SH / lag                                   # per-frame velocity (row,col)
    conf = w.max(1) - w.mean(1)
    return v.astype(np.float32), conf.astype(np.float32)

def zone_features3(S, RC):
    N = S.shape[0]
    X0, n0 = zone_features2(S, RC)
    G4 = to_grid(S, RC)                                # (N,4,4,30)
    G = G4.reshape(N, 16, 30)
    gmean = G.mean((1, 2))[:, None] + 1e-6
    f, names = [], []
    def add(a, n): f.append(np.asarray(a, np.float32)); names.append(n)

    # ---------- optical-flow advection forecast ----------
    for lag in [1, 3]:
        v, conf = flow_vec(G4, lag)
        add(np.broadcast_to(conf[:, None], (N, 16)), f"flowconf{lag}")
        add(np.broadcast_to(np.linalg.norm(v, axis=1)[:, None], (N, 16)), f"flowmag{lag}")
        base = G4[:, :, :, -4:].mean(3)                # recent field
        for off, hn in [(3.5, "h0"), (9.5, "h1"), (15.5, "h2")]:
            pred = _shift_field(base, v * off).reshape(N, 16)
            add(pred / gmean, f"adv{lag}_{hn}")
            add((-pred).argsort(1).argsort(1), f"adv{lag}_{hn}_rank")
        # direction of flow relative to this cell
        cen = (G4.reshape(N, 16, 30)[:, :, -3:].mean(2)[:, :, None] * CR[None]).sum(1) / \
              (G4.reshape(N, 16, 30)[:, :, -3:].mean(2).sum(1)[:, None] + 1e-6)
        rel = CR[None] - cen[:, None, :]
        dist = np.linalg.norm(rel, axis=2) + 1e-6
        add((rel * v[:, None, :]).sum(2) / dist, f"flowproj{lag}")

    # ---------- damped per-cell AR forecast ----------
    m3 = G[:, :, -3:].mean(2); m8 = G[:, :, -8:].mean(2); m20 = G[:, :, -20:].mean(2)
    tr = m3 - G[:, :, -6:-3].mean(2)
    for damp, hn in [(1.0, "h0"), (2.0, "h1"), (3.0, "h2")]:
        pr = m3 + tr * damp * 0.5
        pr = 0.6 * pr + 0.4 * m8                        # shrink toward longer mean
        add(pr / gmean, f"ar_{hn}")
        add((-pr).argsort(1).argsort(1), f"ar_{hn}_rank")
    add((m3 - m20) / gmean, "m3_m20")

    # ---------- leader / alternation structure (multi-peak) ----------
    lead = (G == G.max(1, keepdims=True))
    li = G.argmax(1)                                    # (N,30) leader cell per frame
    nsw = (np.diff(li, axis=1) != 0).mean(1)
    add(np.broadcast_to(nsw[:, None], (N, 16)), "leader_switch_rate")
    # time since this cell last led
    idxT = np.arange(30)[None, None, :]
    lastlead = np.where(lead, idxT, -1).max(2)
    add(29 - lastlead, "frames_since_lead")
    add((lastlead >= 0).astype(np.float32), "ever_led")
    # correlation with the current leader's series (anti-correlated = alternating)
    cur = li[:, -1]
    ls = G[np.arange(N), cur]                           # (N,30)
    lz = (ls - ls.mean(1, keepdims=True)) / (ls.std(1, keepdims=True) + 1e-6)
    gz = (G - G.mean(2, keepdims=True)) / (G.std(2, keepdims=True) + 1e-6)
    add((gz * lz[:, None, :]).mean(2), "corr_with_leader")
    add((G[:, :, -6:].mean(2) - ls[:, -6:].mean(1)[:, None]) / gmean, "gap_to_leader")
    add(np.broadcast_to((np.sort(m3, 1)[:, -1] - np.sort(m3, 1)[:, -2])[:, None] / gmean, (N, 16)), "top1_top2_gap")
    # is this cell adjacent to the leader?
    curpos = CR[cur]
    add(np.abs(CR[None] - curpos[:, None, :]).sum(2), "man_to_leader")

    # ---------- peak-trajectory extrapolation ----------
    pk = CR[li]                                         # (N,30,2) leader position per frame
    t = np.arange(30, dtype=np.float32) - 14.5
    pv = ((pk - pk.mean(1, keepdims=True)) * t[None, :, None]).sum(1) / (t ** 2).sum()
    plast = pk[:, -5:].mean(1)
    for off, hn in [(3.5, "h0"), (9.5, "h1"), (15.5, "h2")]:
        p = plast + pv * off
        add(np.linalg.norm(CR[None] - p[:, None, :], axis=2), f"pkpred_{hn}")
    add(np.broadcast_to(np.linalg.norm(pv, axis=1)[:, None], (N, 16)), "pk_speed")

    X1 = np.stack(f, -1)
    return np.concatenate([X0, X1], -1), n0 + names
