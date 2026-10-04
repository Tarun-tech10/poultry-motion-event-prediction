import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, os
torch.backends.cudnn.benchmark = True
from common import *
from model import Net, make_input, CELLP, DIRP, apply_d4
from scorer import score_arrays
from decode import decode

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EPOCHS = int(os.environ.get("EPOCHS", 60))
BS = int(os.environ.get("BS", 256))
dev = "cuda"
torch.manual_seed(SEED); np.random.seed(SEED)

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z); G = to_grid(S, RC)
X = make_input(G)
cell_of = (RC[:, :, 0].astype(np.int64) * 4 + RC[:, :, 1].astype(np.int64))
ZC = np.take_along_axis(cell_of, Z.astype(np.int64), 1)
cell2alias = np.zeros((N, 16), np.int64)
np.put_along_axis(cell2alias, cell_of, np.arange(16)[None, :].repeat(N, 0), 1)
fold = np.load("fold_group.npy")                      # same group folds as the GBM
FOLDS = fold.max() + 1
print("fold sizes", np.bincount(fold), flush=True)

Xt = torch.tensor(X); ZCt = torch.tensor(ZC); Mt = torch.tensor(M.astype(np.int64)); Dt = torch.tensor(D.astype(np.int64))
CELLPt = torch.tensor(CELLP); DIRPt = torch.tensor(DIRP)
oof_z = np.zeros((N, 3, 16), np.float32); oof_m = np.zeros((N, 3, 4), np.float32); oof_d = np.zeros((N, 3, 9), np.float32)

def run_fold(f):
    tri = np.flatnonzero(fold != f); vai = np.flatnonzero(fold == f)
    net = Net().to(dev); scaler = torch.amp.GradScaler("cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
    spe = max(1, len(tri) // BS); steps = EPOCHS * spe
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps, pct_start=0.3)
    st = 0
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(tri)
        for i in range(0, len(perm) - BS + 1, BS):
            b = perm[i:i + BS]
            t = np.random.randint(8)
            xb = apply_d4(Xt[b].to(dev, non_blocking=True), t)
            zb = CELLPt[t].to(dev)[ZCt[b].to(dev)]
            db = DIRPt[t].to(dev)[Dt[b].to(dev)]
            mb = Mt[b].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                lz, lm, ld = net(xb)
                loss = (F.cross_entropy(lz.reshape(-1, 16), zb.reshape(-1), label_smoothing=0.05)
                        + 0.45 * F.cross_entropy(lm.reshape(-1, 4), mb.reshape(-1), label_smoothing=0.05)
                        + 0.20 * F.cross_entropy(ld.reshape(-1, 9), db.reshape(-1), label_smoothing=0.05))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if st < steps - 1: sch.step(); st += 1
    net.eval()
    pz = np.zeros((len(vai), 3, 16), np.float32); pm = np.zeros((len(vai), 3, 4), np.float32); pd = np.zeros((len(vai), 3, 9), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for t in range(8):
            inv = np.argsort(CELLP[t]); invd = np.argsort(DIRP[t])
            for i in range(0, len(vai), 512):
                b = vai[i:i + 512]
                lz, lm, ld = net(apply_d4(Xt[b].to(dev), t))
                pz[i:i + 512] += F.softmax(lz.float(), -1).cpu().numpy()[:, :, inv] / 8
                pm[i:i + 512] += F.softmax(lm.float(), -1).cpu().numpy() / 8
                pd[i:i + 512] += F.softmax(ld.float(), -1).cpu().numpy()[:, :, invd] / 8
    return vai, pz, pm, pd

t0 = time.time()
for f in range(FOLDS):
    vai, pz, pm, pd = run_fold(f)
    oof_z[vai] = pz; oof_m[vai] = pm; oof_d[vai] = pd
    print(f"fold {f} zone acc {[round(float((pz[:,k].argmax(1)==ZC[vai,k]).mean()),4) for k in range(3)]} ({time.time()-t0:.0f}s)", flush=True)

np.save(f"nn{SEED}_oof_z.npy", oof_z); np.save(f"nn{SEED}_oof_m.npy", oof_m); np.save(f"nn{SEED}_oof_d.npy", oof_d)
print("zone acc", [round(float((oof_z[:,k].argmax(1)==ZC[:,k]).mean()),4) for k in range(3)], flush=True)
print("tier acc", [round(float((oof_m[:,k].argmax(1)==M[:,k]).mean()),4) for k in range(3)], flush=True)
od = np.zeros((N,3,9), np.float32); od[:,:,8]=1.0
for lam in [0.0, 1.0]:
    cells, msel, dsel = decode(oof_z, oof_m, od, lam=lam, force_dc=True)
    al = np.take_along_axis(cell2alias, cells, 1)
    r = score_arrays(al, dsel, msel, Z, D, M, RC)
    print(f"NN lam {lam}: final {r['final']:.4f} ovr {r['overall']:.4f} zone {r['zone']:.3f} tier {r['tier']:.3f}", flush=True)
