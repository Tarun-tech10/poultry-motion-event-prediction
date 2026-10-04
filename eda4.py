import numpy as np, collections
from common import *
from scorer import score_arrays

d = load("../HEN_DATA/train.csv")
S, RC, Z, D, M = d["strip"], d["rc"], d["zone"], d["dirn"], d["tier"]
N = len(Z)

def ev(pZ,pD,pM,lbl):
    r = score_arrays(pZ,pD,pM,Z,D,M,RC)
    print(f"{lbl:38s} final {r['final']:.4f} ovr {r['overall']:.4f} | zone {r['zone']:.3f} dir {r['dirn']:.3f} tier {r['tier']:.3f} trans {r['trans']:.3f} exact {r['exact']:.3f} exl {r['exl']:.4f}")
    return r

am3 = S[:,:,-3:].mean(2).argmax(1)
z_rep = np.repeat(am3[:,None],3,1)
DC = np.full((N,3),8,np.int8)
ev(z_rep, DC, np.full((N,3),2,np.int8), "argmax3 / DC / M2")
ev(z_rep, DC, np.full((N,3),1,np.int8), "argmax3 / DC / M1")
# oracle-ish upper bounds
ev(Z, DC, np.full((N,3),2,np.int8), "ORACLE zone / DC / M2")
ev(Z, D, M, "FULL ORACLE")
ev(z_rep, D, M, "argmax3 zone / oracle dir+tier")
ev(Z, DC, M, "oracle zone+tier / DC")

# what does always-DC give vs best fixed dir
for dd in range(9):
    r=score_arrays(z_rep,np.full((N,3),dd,np.int8),np.full((N,3),2,np.int8),Z,D,M,RC)
    print("  fixed dir",ALLD[dd],"->",round(r['final'],4), "dirscore",round(r['dirn'],4))

print("\n=== best single tier per horizon ===")
for k in range(3):
    print(" H%d tier counts"%k, collections.Counter(M[:,k].tolist()).most_common())

print("\n=== zone predictability ceiling probes ===")
for w in [1,2,3,4,5,8]:
    a = S[:,:,-w:].mean(2).argmax(1)
    print(f" last{w}: H0 {(a==Z[:,0]).mean():.4f} H1 {(a==Z[:,1]).mean():.4f} H2 {(a==Z[:,2]).mean():.4f}")
# weighted recency
for tau in [1.5,3,5,8,15]:
    w = np.exp(-np.arange(29,-1,-1)/tau)[None,None,:]
    a = (S*w).sum(2).argmax(1)
    print(f" tau{tau}: H0 {(a==Z[:,0]).mean():.4f} H1 {(a==Z[:,1]).mean():.4f} H2 {(a==Z[:,2]).mean():.4f}")

# top-k coverage of truth by last3 ranking
rank = np.argsort(-S[:,:,-3:].mean(2),1)
for k in [1,2,3,5]:
    cov = np.array([[Z[i,h] in rank[i,:k] for h in range(3)] for i in range(N)]).mean(0)
    print(f" top{k} coverage H0 {cov[0]:.3f} H1 {cov[1]:.3f} H2 {cov[2]:.3f}")
