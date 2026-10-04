"""Lattice GNN over the 4x4 grid: per-cell temporal encoder + message passing, D4-augmented.
Small and CPU-viable by design (no Conv3d). Evaluated on the BALANCED objective."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, os, sys
from common import *
from scorer import score_arrays
from decode import decode
from evalbal import rule_families, balanced_eval, NAMES

SEED = int(os.environ.get("SEED", 0)); EPOCHS = int(os.environ.get("EPOCHS", 400))
DEV = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(os.cpu_count() or 8)

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC).reshape(N, 16, 30)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
c2a = np.zeros((N, 16), np.int64); np.put_along_axis(c2a, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)
fold = np.load("fold_group.npy"); fam = rule_families(ZC, M)
CRc = np.array([[i // 4, i % 4] for i in range(16)])
ADJ = torch.tensor((np.abs(CRc[:, None] - CRc[None, :]).sum(-1) == 1).astype(np.float32))
ADJ = ADJ / ADJ.sum(1, keepdim=True)

# D4 permutations of the 16 cells
def d4perm(t):
    g = np.arange(16).reshape(4, 4)
    if t & 4: g = g.T
    g = np.rot90(g, t & 3)
    return g.reshape(-1)
PERM = np.stack([d4perm(t) for t in range(8)])          # PERM[t][new_cell] = old_cell
INV = np.stack([np.argsort(p) for p in PERM])

# inputs: per-row normalised motion + D4-invariant node descriptors
rowmean = G.mean((1, 2), keepdims=True) + 1e-6
Xt = np.stack([G / rowmean, np.log1p(G) - np.log1p(G.mean((1, 2), keepdims=True))], 1).astype(np.float32)  # (N,2,16,30)
node = np.stack([np.abs(CRc - 1.5).sum(1) / 3.0, (np.abs(CRc[:, None] - CRc[None, :]).sum(-1) == 1).sum(1) / 4.0], 1).astype(np.float32)
NODE = torch.tensor(node)                                # (16,2) D4-invariant

class Net(nn.Module):
    def __init__(s, dm=96):
        super().__init__()
        s.tenc = nn.Sequential(
            nn.Conv1d(2, 48, 5, padding=2), nn.GELU(),
            nn.Conv1d(48, 64, 5, padding=2, dilation=1), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.GELU())
        s.proj = nn.Linear(64 * 3 + 2, dm)
        s.mp = nn.ModuleList([nn.Linear(dm * 3, dm) for _ in range(3)])
        s.ln = nn.ModuleList([nn.LayerNorm(dm) for _ in range(3)])
        s.zhead = nn.Linear(dm, 3)
        s.mhead = nn.Sequential(nn.Linear(dm * 2, 128), nn.GELU(), nn.Linear(128, 12))
    def forward(s, x):                                   # x: (B,2,16,30)
        B = x.shape[0]
        h = s.tenc(x.permute(0, 2, 1, 3).reshape(B * 16, 2, 30))     # (B*16,64,30)
        h = torch.cat([h[:, :, -1], h[:, :, -6:].mean(2), h.mean(2)], 1)
        h = h.reshape(B, 16, -1)
        h = s.proj(torch.cat([h, NODE.to(x.device).expand(B, 16, 2)], 2))
        A = ADJ.to(x.device)
        for lin, ln in zip(s.mp, s.ln):
            nb = torch.einsum("ij,bjd->bid", A, h)
            g = h.mean(1, keepdim=True).expand_as(h)
            h = ln(h + F.gelu(lin(torch.cat([h, nb, g], 2))))
        z = s.zhead(h).permute(0, 2, 1)                                # (B,3,16)
        pooled = torch.cat([h.mean(1), h.max(1).values], 1)
        return z, s.mhead(pooled).reshape(B, 3, 4)

Xtt = torch.tensor(Xt); ZCt = torch.tensor(ZC); Mt = torch.tensor(M.astype(np.int64))
PERMt = torch.tensor(PERM)
oz = np.zeros((N, 3, 16), np.float32); om = np.zeros((N, 3, 4), np.float32)
t0 = time.time()
for f in range(5):
    tri = np.flatnonzero(fold != f); vai = np.flatnonzero(fold == f)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    BS = 256; spe = max(1, len(tri) // BS); steps = EPOCHS * spe
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps, pct_start=0.25)
    st = 0
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(tri)
        for i in range(0, len(perm) - BS + 1, BS):
            b = perm[i:i + BS]
            t = np.random.randint(8)
            xb = Xtt[b][:, :, PERM[t], :].to(DEV)
            zb = torch.tensor(INV[t])[ZCt[b]].to(DEV)
            mb = Mt[b].to(DEV)
            zl, ml = net(xb)
            loss = (F.cross_entropy(zl.reshape(-1, 16), zb.reshape(-1), label_smoothing=0.05)
                    + 0.4 * F.cross_entropy(ml.reshape(-1, 4), mb.reshape(-1), label_smoothing=0.05))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if st < steps - 1: sch.step(); st += 1
    net.eval()
    with torch.no_grad():
        pz = np.zeros((len(vai), 3, 16), np.float32); pm = np.zeros((len(vai), 3, 4), np.float32)
        for t in range(8):                                  # D4 TTA
            for i in range(0, len(vai), 512):
                b = vai[i:i + 512]
                zl, ml = net(Xtt[b][:, :, PERM[t], :].to(DEV))
                pz[i:i + 512] += F.softmax(zl.float(), -1).cpu().numpy()[:, :, PERM[t]] / 8
                pm[i:i + 512] += F.softmax(ml.float(), -1).cpu().numpy() / 8
    oz[vai] = pz; om[vai] = pm
    print(f"fold {f} zone {[round(float((pz[:,k].argmax(1)==ZC[vai,k]).mean()),4) for k in range(3)]} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"gnn{SEED}_oof_z.npy", oz); np.save(f"gnn{SEED}_oof_m.npy", om)
print("GNN zone acc", [round(float((oz[:,k].argmax(1)==ZC[:,k]).mean()),4) for k in range(3)], flush=True)
print("GNN tier acc", [round(float((om[:,k].argmax(1)==M[:,k]).mean()),4) for k in range(3)], flush=True)

ez = np.load("ens_oof_z.npy"); em = np.load("ens_oof_m.npy")
od = np.zeros((N, 3, 9), np.float32); od[:, :, 8] = 1.0
def ev(pz, pm, lam=1.5, T=0.9):
    p = pz ** (1 / T); p /= p.sum(2, keepdims=True)
    cells, msel, dsel = decode(p, pm, od, lam=lam, force_dc=True)
    rows = score_arrays(np.take_along_axis(c2a, cells, 1), dsel, msel, Z, D, M, RC, reduce=False)
    return balanced_eval(rows, fam)
print(f"\nGBM alone   balanced {ev(ez,em)[0]:.4f}")
print(f"GNN alone   balanced {ev(oz,om)[0]:.4f}")
for w in [0.15, 0.25, 0.35, 0.5]:
    lo = (1 - w) * np.log(np.clip(ez, 1e-7, 1)) + w * np.log(np.clip(oz, 1e-7, 1))
    e = np.exp(lo - lo.max(2, keepdims=True)); bz = (e / e.sum(2, keepdims=True)).astype(np.float32)
    bm = (1 - w) * em + w * om
    r = ev(bz, bm)
    print(f"blend w={w:.2f} balanced {r[0]:.4f} (ovr {r[1]:.4f} worst {r[2]:.4f})  zoneacc "
          f"{[round(float((bz[:,k].argmax(1)==ZC[:,k]).mean()),4) for k in range(3)]}")
