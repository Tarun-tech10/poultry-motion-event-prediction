"""Shared train/predict pipeline: zone ensemble + marginalised tier, used by CV and by solution.py."""
import numpy as np, lightgbm as lgb

ZCFG = [
    dict(learning_rate=0.030, num_leaves=63, min_data_in_leaf=80, feature_fraction=0.45,
         bagging_fraction=0.80, lambda_l2=8.0, seed=1, rounds=320),
    dict(learning_rate=0.035, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.65,
         bagging_fraction=0.70, lambda_l2=3.0, seed=2, rounds=300),
    dict(learning_rate=0.025, num_leaves=127, min_data_in_leaf=120, feature_fraction=0.35,
         bagging_fraction=0.85, lambda_l2=15.0, seed=3, rounds=340),
    dict(learning_rate=0.040, num_leaves=48, min_data_in_leaf=60, feature_fraction=0.55,
         bagging_fraction=0.75, lambda_l2=6.0, seed=4, rounds=260),
]
MCFG = dict(objective="multiclass", num_class=4, learning_rate=0.035, num_leaves=31,
            min_data_in_leaf=60, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=5.0, verbose=-1, max_bin=127)
TOPC = 6          # candidate cells for tier marginalisation

def _p(cfg, threads):
    return dict(objective="binary", verbose=-1, num_threads=threads, max_bin=127, bagging_freq=1,
                learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
                min_data_in_leaf=cfg["min_data_in_leaf"], feature_fraction=cfg["feature_fraction"],
                bagging_fraction=cfg["bagging_fraction"], lambda_l2=cfg["lambda_l2"], seed=cfg["seed"])

RANK_W, RANK_T = 0.35, 0.75
RCFG = dict(learning_rate=0.03, num_leaves=63, min_data_in_leaf=80, feature_fraction=0.45,
            bagging_fraction=0.8, lambda_l2=8.0, seed=1, rounds=320)

def fit_zone(XZtr, ZCtr, XZte, threads=8, cfgs=ZCFG):
    """XZ*: (N,16,F). Log-space blend of binary-relevance ensemble + a lambdarank member."""
    Ntr, _, F = XZtr.shape; Nte = XZte.shape[0]
    Ftr = XZtr.reshape(-1, F); Fte = XZte.reshape(-1, F)
    out = np.zeros((Nte, 3, 16), np.float32)
    for k in range(3):
        y = np.zeros((Ntr, 16), np.float32); y[np.arange(Ntr), ZCtr[:, k]] = 1
        acc = np.zeros((Nte, 16), np.float64)
        for cfg in cfgs:
            ds = lgb.Dataset(Ftr, y.reshape(-1))
            m = lgb.train(_p(cfg, threads), ds, cfg["rounds"])
            acc += np.log(np.clip(m.predict(Fte).reshape(Nte, 16), 1e-7, 1))
        acc /= len(cfgs)
        rp = dict(_p(RCFG, threads), objective="lambdarank", lambdarank_truncation_level=4)
        rm = lgb.train(rp, lgb.Dataset(Ftr, y.reshape(-1), group=np.full(Ntr, 16)), RCFG["rounds"])
        rs = rm.predict(Fte).reshape(Nte, 16) / RANK_T
        rs = rs - rs.max(1, keepdims=True)
        rlog = rs - np.log(np.exp(rs).sum(1, keepdims=True))
        acc = (1 - RANK_W) * acc + RANK_W * rlog
        e = np.exp(acc - acc.max(1, keepdims=True))
        out[:, k] = (e / e.sum(1, keepdims=True)).astype(np.float32)
    return out

def fit_tier(XZtr, XRtr, Mtr, ZCtr, XZte, XRte, pz_te, threads=8, rounds=420):
    """Tier conditioned on the winning cell, then marginalised over the zone distribution."""
    Nte = XZte.shape[0]
    out = np.zeros((Nte, 3, 4), np.float32)
    cand = np.argsort(-pz_te, axis=2)[:, :, :TOPC]                      # (Nte,3,TOPC)
    for k in range(3):
        Xtr = np.concatenate([XRtr, XZtr[np.arange(len(XZtr)), ZCtr[:, k]]], 1)
        m = lgb.train(dict(MCFG, num_threads=threads, seed=7), lgb.Dataset(Xtr, Mtr[:, k]), rounds)
        num = np.zeros((Nte, 4)); den = np.zeros((Nte, 1))
        for j in range(TOPC):
            c = cand[:, k, j]
            w = pz_te[np.arange(Nte), k, c][:, None]
            Xte = np.concatenate([XRte, XZte[np.arange(Nte), c]], 1)
            num += w * m.predict(Xte); den += w
        out[:, k] = (num / np.maximum(den, 1e-9)).astype(np.float32)
    return out
