import json, numpy as np, pandas as pd

ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
IDX = np.full(256, -1, np.int16)
for i, c in enumerate(ALPHA):
    IDX[ord(c)] = i
DIRS = ["DN","DNE","DE","DSE","DS","DSW","DW","DNW"]
ALLD = DIRS + ["DC"]
# compass unit vectors in (drow, dcol) with row increasing = "south"?? test both
DVEC = {"DN":(-1,0),"DNE":(-1,1),"DE":(0,1),"DSE":(1,1),"DS":(1,0),"DSW":(1,-1),"DW":(0,-1),"DNW":(-1,-1)}

def decode_strip(s):
    a = IDX[np.frombuffer(s.encode(), np.uint8)]
    return a.reshape(16, 30).astype(np.float32)

def load(path, with_target=True):
    df = pd.read_csv(path)
    N = len(df)
    strips = np.zeros((N, 16, 30), np.float32)
    rc = np.zeros((N, 16, 2), np.int8)   # zone index -> (grid_row, grid_col)
    for i, (s, zc) in enumerate(zip(df.past_motion_strip.values, df.zone_cards.values)):
        strips[i] = decode_strip(s)
        cards = json.loads(zc)
        cards.sort(key=lambda z: z["zone"])
        for j, z in enumerate(cards):
            assert int(z["zone"][1:]) == j + 1
            rc[i, j] = (z["grid_row"], z["grid_col"])
    out = {"id": df.id.values, "strip": strips, "rc": rc}
    if with_target:
        Z = np.zeros((N, 3), np.int8); D = np.zeros((N, 3), np.int8); M = np.zeros((N, 3), np.int8)
        for i, t in enumerate(df.target_future_ledger.values):
            for k, tok in enumerate(t.split()):
                h, z, d, m = tok.split(":")
                assert h == "H%d" % k
                Z[i, k] = int(z[1:]) - 1
                D[i, k] = ALLD.index(d)
                M[i, k] = int(m[1:])
        out.update(zone=Z, dirn=D, tier=M)
    return out

def to_grid(strip, rc):
    """(N,16,T) row-local -> (N,4,4,T) spatial grid."""
    N, _, T = strip.shape
    g = np.zeros((N, 4, 4, T), np.float32)
    idx = np.arange(N)[:, None]
    g[idx, rc[:, :, 0], rc[:, :, 1]] = strip
    return g
