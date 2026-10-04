import numpy as np
from common import ALLD

def score_arrays(pZ, pD, pM, tZ, tD, tM, RC, families=None, reduce=True):
    """pZ,tZ: (N,3) zone idx 0..15; pD,tD: (N,3) dir idx 0..8; pM,tM:(N,3) tier 0..3
       RC: (N,16,2) grid coords. Returns final_score components."""
    N = len(pZ); ar = np.arange(N)[:, None]
    prc = np.stack([RC[ar, pZ, 0], RC[ar, pZ, 1]], -1).astype(int)
    trc = np.stack([RC[ar, tZ, 0], RC[ar, tZ, 1]], -1).astype(int)
    man = np.abs(prc - trc).sum(-1)
    zs = np.where(pZ == tZ, 1.0, np.where(man == 1, 0.1, 0.0))

    cd = np.minimum((pD - tD) % 8, (tD - pD) % 8)
    both = (pD < 8) & (tD < 8)
    ds = np.where(pD == tD, 1.0, np.where(both & (cd == 1), 0.1, 0.0))

    td = np.abs(pM.astype(int) - tM.astype(int))
    ts = np.where(td == 0, 1.0, np.where(td == 1, 0.1, 0.0))

    ex = ((pZ == tZ) & (pD == tD) & (pM == tM)).astype(float)
    ev = 0.52 * zs + 0.24 * ds + 0.14 * ts + 0.10 * ex

    big = ((pZ[:, :2] == tZ[:, :2]) & (pZ[:, 1:] == tZ[:, 1:])).astype(float)
    trans = big.mean(1)
    exl = (ex.sum(1) == 3).astype(float)
    row = 0.64 * ev.mean(1) + 0.18 * trans + 0.18 * exl
    if not reduce:
        return row
    om = row.mean()
    b20 = np.sort(row)[: max(1, int(round(0.2 * N)))].mean()
    if families is not None:
        wf = min(row[families == f].mean() for f in np.unique(families))
    else:
        wf = om
    return dict(final=0.70 * om + 0.18 * wf + 0.12 * b20, overall=om, worst=wf, bot20=b20,
                zone=zs.mean(), dirn=ds.mean(), tier=ts.mean(), exact=ex.mean(), trans=trans.mean(), exl=exl.mean())
