import numpy as np, lightgbm as lgb, time
from common import *
from feats3 import zone_features3
d=load('../HEN_DATA/train.csv'); S,RC,Z=d['strip'],d['rc'],d['zone']
N=len(Z); cell_of=(RC[:,:,0].astype(np.int64)*4+RC[:,:,1].astype(np.int64))
ZC=np.take_along_axis(cell_of,Z.astype(np.int64),1)
X,_=zone_features3(S,RC); F=X.shape[2]; Xf=X.reshape(-1,F)
fold=np.load('fold_group.npy')
def run(obj):
    acc=np.zeros(3); oof=np.zeros((N,3,16),np.float32)
    for k in range(3):
        y=np.zeros((N,16),np.float32); y[np.arange(N),ZC[:,k]]=1; yf=y.reshape(-1)
        for f in range(5):
            trm=fold!=f; vam=fold==f
            tri=np.repeat(trm,16); vai=np.repeat(vam,16)
            if obj=='rank':
                p=dict(objective='lambdarank',lambdarank_truncation_level=4,learning_rate=0.03,
                       num_leaves=63,min_data_in_leaf=80,feature_fraction=0.45,bagging_fraction=0.8,
                       bagging_freq=1,lambda_l2=8.0,verbose=-1,num_threads=6,max_bin=127,seed=1)
                ds=lgb.Dataset(Xf[tri],yf[tri],group=np.full(trm.sum(),16))
            else:
                p=dict(objective='binary',learning_rate=0.03,num_leaves=63,min_data_in_leaf=80,
                       feature_fraction=0.45,bagging_fraction=0.8,bagging_freq=1,lambda_l2=8.0,
                       verbose=-1,num_threads=6,max_bin=127,seed=1)
                ds=lgb.Dataset(Xf[tri],yf[tri])
            m=lgb.train(p,ds,320)
            oof[vam,k]=m.predict(Xf[vai]).reshape(-1,16)
        acc[k]=(oof[:,k].argmax(1)==ZC[:,k]).mean()
    return acc,oof
for obj in ['binary','rank']:
    t=time.time(); a,o=run(obj); print(obj,np.round(a,4),f'{time.time()-t:.0f}s',flush=True)
    np.save(f'zone_{obj}.npy',o)
