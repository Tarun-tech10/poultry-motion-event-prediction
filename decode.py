import numpy as np
from itertools import product

CR = np.array([[r, c] for r in range(4) for c in range(4)])
MAN = np.abs(CR[:, None, :] - CR[None, :, :]).sum(-1)
ADJ = (MAN == 1).astype(np.float32)            # 16x16
TADJ = np.zeros((4, 4), np.float32)
for a in range(4):
    for b in range(4):
        TADJ[a, b] = 1.0 if a == b else (0.1 if abs(a - b) == 1 else 0.0)
DADJ = np.zeros((9, 9), np.float32)
for a in range(9):
    for b in range(9):
        if a == b: DADJ[a, b] = 1.0
        elif a < 8 and b < 8 and min((a - b) % 8, (b - a) % 8) == 1: DADJ[a, b] = 0.1

TRIP = np.array(list(product(range(16), repeat=3)))   # 4096,3

def decode(pz, pm, pd, lam=0.0, w_trans=0.18, force_dc=False, beta=0.0, beta_m=0.0):
    """pz:(N,3,16) cell probs, pm:(N,3,4), pd:(N,3,9). Returns cells(N,3), tier(N,3), dirs(N,3).
    beta/beta_m add risk aversion: reward P(nonzero credit), which lifts the bottom-20% term."""
    N = pz.shape[0]
    Zs = pz + 0.1 * (pz @ ADJ.T)                       # (N,3,16) expected ZoneScore
    Zn = pz + (pz @ ADJ.T)                             # P(zone credit > 0)
    Ts = pm @ TADJ.T                                   # (N,3,4)
    Tn = pm @ (TADJ > 0).astype(np.float32).T          # P(tier credit > 0)
    Ds = pd @ DADJ.T                                   # (N,3,9)
    if force_dc:
        dsel = np.full((N, 3), 8, np.int64)
    else:
        dsel = Ds.argmax(2)
    msel = (Ts + beta_m * Tn).argmax(2)
    ar = np.arange(N)[:, None]; hh = np.arange(3)[None, :]
    pd_s = pd[ar, hh, dsel]; pm_s = pm[ar, hh, msel]    # (N,3)
    base = (0.64 / 3) * (0.52 * Zs + 0.10 * pz * pd_s[:, :, None] * pm_s[:, :, None]) + beta * Zn
    a0 = base[:, 0][:, TRIP[:, 0]]
    a1 = base[:, 1][:, TRIP[:, 1]]
    a2 = base[:, 2][:, TRIP[:, 2]]
    p0 = pz[:, 0][:, TRIP[:, 0]]; p1 = pz[:, 1][:, TRIP[:, 1]]; p2 = pz[:, 2][:, TRIP[:, 2]]
    eq01 = (TRIP[:, 0] == TRIP[:, 1])[None, :]; eq12 = (TRIP[:, 1] == TRIP[:, 2])[None, :]
    j01 = p0 * p1 * (1.0 + lam * eq01); j12 = p1 * p2 * (1.0 + lam * eq12)
    obj = a0 + a1 + a2 + (w_trans / 2.0) * (j01 + j12)
    best = obj.argmax(1)
    cells = TRIP[best]
    return cells, msel, dsel
