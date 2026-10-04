import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import DVEC, ALLD

# ---------------- D4 group on the 4x4 grid (cells indexed r*4+c) ----------------
def d4_cell_perm(t):
    """CELLP[t][j] = new index of content originally at cell j."""
    fr, fc, tp = t & 1, (t >> 1) & 1, (t >> 2) & 1
    p = np.zeros(16, np.int64)
    for r in range(4):
        for c in range(4):
            r1, c1 = (3 - r if fr else r), (3 - c if fc else c)
            if tp: r1, c1 = c1, r1
            p[r * 4 + c] = r1 * 4 + c1
    return p

def d4_dir_perm(t):
    fr, fc, tp = t & 1, (t >> 1) & 1, (t >> 2) & 1
    p = np.zeros(9, np.int64); p[8] = 8
    for i, name in enumerate(ALLD[:8]):
        dr, dc = DVEC[name]
        dr1, dc1 = (-dr if fr else dr), (-dc if fc else dc)
        if tp: dr1, dc1 = dc1, dr1
        p[i] = [j for j, n2 in enumerate(ALLD[:8]) if DVEC[n2] == (dr1, dc1)][0]
    return p

CELLP = np.stack([d4_cell_perm(t) for t in range(8)])
DIRP  = np.stack([d4_dir_perm(t) for t in range(8)])
GATHER = np.stack([np.argsort(CELLP[t]) for t in range(8)])   # x_new = x[..., GATHER[t], :]

CR = np.array([[r, c] for r in range(4) for c in range(4)])
_MAN = np.abs(CR[:, None, :] - CR[None, :, :]).sum(-1)
A1 = (_MAN == 1).astype(np.float32)
A2 = (_MAN == 2).astype(np.float32)

def apply_d4(x, t):
    """x: (B,C,16,T) -> spatially transformed"""
    return x[:, :, torch.as_tensor(GATHER[t], device=x.device), :]

# ---------------- model ----------------
class SpatialMix(nn.Module):
    """per-channel mixing over the 4x4 lattice using distance-1/2 adjacency (D4-equivariant)."""
    def __init__(s, c):
        super().__init__()
        B = np.stack([np.eye(16, dtype=np.float32), A1, A2])   # (3,16,16)
        s.register_buffer("B", torch.tensor(B))
        s.w = nn.Parameter(torch.stack([torch.ones(c), torch.zeros(c), torch.zeros(c)]))
    def forward(s, x):                                        # (B,C,16,T)
        M = torch.einsum("kc,kij->cij", s.w, s.B)             # (C,16,16)
        return torch.einsum("bcjt,cij->bcit", x, M.to(x.dtype))

class Block(nn.Module):
    """depthwise temporal conv + lattice mix + pointwise channel mix."""
    def __init__(s, c, k=5, dil=1):
        super().__init__()
        s.tc = nn.Conv2d(c, c, (1, k), padding=(0, dil * (k - 1) // 2), dilation=(1, dil), groups=c)
        s.bn = nn.BatchNorm2d(c)
        s.sm = SpatialMix(c)
        s.pw = nn.Conv2d(c, c, 1); s.bn2 = nn.BatchNorm2d(c)
        s.pw2 = nn.Conv2d(c, c, 1); s.bn3 = nn.BatchNorm2d(c)
    def forward(s, x):
        h = F.gelu(s.bn(s.tc(x)))
        h = F.gelu(s.bn2(s.pw(s.sm(h))))
        h = s.bn3(s.pw2(h))
        return F.gelu(x + h)

class Net(nn.Module):
    """progressive time downsampling 30 -> 15 -> 5 keeps activations small."""
    def __init__(s, cin=4, c=80, dils=(1, 2, 4, 1, 2, 1)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(cin, c, (1, 5), padding=(0, 2)), nn.BatchNorm2d(c), nn.GELU())
        s.b1 = nn.ModuleList([Block(c, 5, d) for d in dils[:2]])
        s.d1 = nn.Sequential(nn.Conv2d(c, c, (1, 4), stride=(1, 2), padding=(0, 1)), nn.BatchNorm2d(c), nn.GELU())
        s.b2 = nn.ModuleList([Block(c, 5, d) for d in dils[2:4]])
        s.d2 = nn.Sequential(nn.Conv2d(c, c, (1, 5), stride=(1, 3), padding=(0, 1)), nn.BatchNorm2d(c), nn.GELU())
        s.b3 = nn.ModuleList([Block(c, 3, d) for d in dils[4:]])
        s.tcollapse = nn.Conv2d(c, c, (1, 5))                 # learned temporal pooling (T=5)
        s.bnT = nn.BatchNorm2d(c)
        s.post = nn.ModuleList([SpatialMix(3 * c), SpatialMix(c)])
        s.pw1 = nn.Conv1d(3 * c, c, 1); s.bnp = nn.BatchNorm1d(c)
        s.zone = nn.Conv1d(c, 3, 1)
        s.gp = nn.Sequential(nn.Linear(2 * c, 256), nn.GELU(), nn.Dropout(0.1))
        s.tier = nn.Linear(256, 12); s.dir = nn.Linear(256, 27)
    def forward(s, x):                                        # (B,cin,16,30)
        h = s.stem(x)
        for b in s.b1: h = b(h)
        h30 = h
        h = s.d1(h)
        for b in s.b2: h = b(h)
        h = s.d2(h)
        for b in s.b3: h = b(h)
        a = F.gelu(s.bnT(s.tcollapse(h)))[:, :, :, 0]         # (B,c,16)
        m = h.mean(3); mx = h30[:, :, :, -6:].mean(3)         # recency-focused branch
        f = torch.cat([a, m, mx], 1)                          # (B,3c,16)
        f = s.post[0](f.unsqueeze(-1)).squeeze(-1)
        f = F.gelu(s.bnp(s.pw1(f)))
        f = s.post[1](f.unsqueeze(-1)).squeeze(-1)
        z = s.zone(f)                                         # (B,3,16)
        g = torch.cat([f.mean(2), f.amax(2)], 1)
        g = s.gp(g)
        return z, s.tier(g).view(-1, 3, 4), s.dir(g).view(-1, 3, 9)

def make_input(G):
    """G: (N,4,4,30) -> (N,C,16,30)"""
    N = G.shape[0]
    S = G.reshape(N, 16, 30)
    x = S / 16.0
    mu = S.mean((1, 2), keepdims=True); sd = S.std((1, 2), keepdims=True) + 1e-6
    xn = (S - mu) / sd
    glob = np.broadcast_to(S.mean(1)[:, None, :], (N, 16, 30)) / 16.0
    xc = (S - S.mean(2, keepdims=True)) / (S.std(2, keepdims=True) + 1e-6)
    return np.stack([x, xn, glob, xc], 1).astype(np.float32)
