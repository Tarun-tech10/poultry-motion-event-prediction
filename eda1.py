import json, numpy as np, pandas as pd, collections

ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
IDX = {c: i for i, c in enumerate(ALPHA)}
DIRS = ["DN","DNE","DE","DSE","DS","DSW","DW","DNW"]

tr = pd.read_csv("../HEN_DATA/train.csv")
te = pd.read_csv("../HEN_DATA/test.csv")
print(tr.shape, te.shape)
print(tr.columns.tolist())

def decode(s):
    a = np.array([IDX[c] for c in s], dtype=np.int16)
    return a.reshape(16, 30)

r0 = tr.iloc[0]
S = decode(r0.past_motion_strip)
print("strip stats", S.min(), S.max(), S.mean())

zc = json.loads(r0.zone_cards)
print(len(zc))
for z in zc[:3]: print(z)

# check zone ordering: are cards sorted by zone alias?
zones = [z["zone"] for z in zc]
print("sorted?", zones == sorted(zones))

# check grid positions form a full 4x4 permutation
pos = sorted((z["grid_row"], z["grid_col"]) for z in zc)
print("full grid?", pos == [(r,c) for r in range(4) for c in range(4)])

# check neighbors consistency with grid adjacency
pmap = {z["zone"]: (z["grid_row"], z["grid_col"]) for z in zc}
bad = 0
for z in zc:
    r,c = pmap[z["zone"]]
    true_nb = {zz for zz,(rr,cc) in pmap.items() if abs(rr-r)+abs(cc-c)==1}
    if set(z["neighbors"]) != true_nb: bad += 1
print("neighbor mismatch count", bad, "example card nb", zc[0]["neighbors"])

# row_band / col_band vs grid_row/col
rb = collections.Counter((z["row_band"], z["grid_row"]) for z in zc)
cb = collections.Counter((z["col_band"], z["grid_col"]) for z in zc)
print("row_band map", sorted(rb))
print("col_band map", sorted(cb))

# targets
print(tr.target_future_ledger.head(5).tolist())
print("consts:", tr.strip_shape.unique(), tr.frame_rate_hz.unique(), tr.forecast_horizons.unique(), tr.max_events.unique())

def parse(t):
    out = []
    for tok in t.split():
        h,z,d,m = tok.split(":")
        out.append((h,z,d,m))
    return out

P = [parse(t) for t in tr.target_future_ledger]
zs = collections.Counter(x[1] for p in P for x in p)
ds = collections.Counter(x[2] for p in P for x in p)
ms = collections.Counter(x[3] for p in P for x in p)
print("zone dist", sorted(zs.items()))
print("dir dist", sorted(ds.items(), key=lambda kv:-kv[1]))
print("tier dist", sorted(ms.items()))

# per-horizon dir/tier dist
for k in range(3):
    print("H%d dir"%k, collections.Counter(p[k][2] for p in P).most_common())
    print("H%d tier"%k, collections.Counter(p[k][3] for p in P).most_common())
