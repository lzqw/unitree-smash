#!/usr/bin/env python3
"""Fit a time-partitioned linear feedback policy from measured teacher actions.

Both clock boundaries and action coefficients are learned from training data.
No teacher parameters, action template, or future trajectory are evaluated.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.control.piecewise_affine import FEATURES,COMPOSED_FEATURES,PiecewiseAffinePolicy,policy_features

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--max-leaves',type=int,default=32)
parser.add_argument('--composed',action='store_true')
parser.add_argument('--start-time',type=float,default=0.)
parser.add_argument('--end-time',type=float,default=float('inf'))
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
data=torch.load(args.dataset,map_location='cpu',weights_only=False)
if not data.get('homogeneous_controller',True):raise ValueError('Freeze and replay one selected teacher before fitting a clock policy')
mask=(data['observations'][:,0,775]>=args.start_time)&(data['observations'][:,0,775]<=args.end_time)
observations=data['observations'][mask].clone()
observations[:,:,775]-=args.start_time
actions=data['actions'][mask]
if len(observations)<2:raise ValueError('Too few selected control steps')
features=COMPOSED_FEATURES if args.composed else FEATURES
transform='body-pitch-clock-v1' if args.composed else 'none'
extracted=policy_features(observations.reshape(-1,794),torch.tensor(features),transform)
o=extracted.reshape(*observations.shape[:2],-1).numpy()
y=actions.numpy()
hold=np.arange(o.shape[1])%5==0
raw=o[:,~hold].reshape(-1,o.shape[2]).astype(np.float64)
target=y[:,~hold].reshape(-1,18).astype(np.float64)
mean=raw.mean(0)
scale=raw.std(0).clip(.01)
x=np.c_[np.ones(len(raw)),(raw-mean)/scale]
times=np.unique(raw[:,0])
bins=np.searchsorted(times,raw[:,0])
n=len(times)
# Diagonal joint damping plus body-pitch feedback. Arm outputs use time.
# Samples exactly at action clipping limits are censored observations of the
# underlying affine command and are excluded from its least-squares fit.
selectors=([[0,1,2,3,4,5,6+j] if j<12 else [0,1,*range(len(features)+1,len(features)+5)] for j in range(18)]
           if args.composed else [[0,1,2,3+j] if j<12 else [0,1] for j in range(18)])
width=max(map(len,selectors))
xx=np.zeros((n,18,width,width))
xy=np.zeros((n,18,width))
yy=np.zeros((n,18))
all_y=np.zeros((n,18))
all_yy=np.zeros((n,18))
all_count=np.zeros(n)
for tick in range(n):
    rows=bins==tick
    all_y[tick]=target[rows].sum(0)
    all_yy[tick]=np.square(target[rows]).sum(0)
    all_count[tick]=int(rows.sum())
    for joint,columns in enumerate(selectors):
        eligible=rows & (np.abs(target[:,joint])<.99999)
        a=np.zeros((int(eligible.sum()),width))
        a[:,:len(columns)]=x[eligible][:,columns]
        b=target[eligible,joint]
        xx[tick,joint]=a.T@a
        xy[tick,joint]=a.T@b
        yy[tick,joint]=b@b
xx=np.concatenate((np.zeros_like(xx[:1]),np.cumsum(xx,axis=0)))
xy=np.concatenate((np.zeros_like(xy[:1]),np.cumsum(xy,axis=0)))
yy=np.concatenate((np.zeros_like(yy[:1]),np.cumsum(yy,axis=0)))
all_y=np.concatenate((np.zeros_like(all_y[:1]),np.cumsum(all_y,axis=0)))
all_yy=np.concatenate((np.zeros_like(all_yy[:1]),np.cumsum(all_yy,axis=0)))
all_count=np.r_[0.,np.cumsum(all_count)]
cache={}

def fit(start,end):
    key=(start,end)
    if key not in cache:
        weights=np.zeros((x.shape[1],18))
        error=0.
        for joint,columns in enumerate(selectors):
            size=len(columns)
            a=(xx[end,joint]-xx[start,joint])[:size,:size]
            b=(xy[end,joint]-xy[start,joint])[:size]
            if a[0,0]<.5:
                coefficient=np.zeros(size)
                coefficient[0]=(all_y[end,joint]-all_y[start,joint])/(all_count[end]-all_count[start])
                error+=max(0.,float(all_yy[end,joint]-all_yy[start,joint]-(all_count[end]-all_count[start])*coefficient[0]**2))
            else:
                coefficient=np.linalg.lstsq(a+np.eye(size)*1e-12,b,rcond=1e-12)[0]
            error+=max(0.,float(yy[end,joint]-yy[start,joint]-2*coefficient@b+coefficient@a@coefficient))
            weights[columns,joint]=coefficient
        cache[key]=(error,weights)
    return cache[key]

leaves=[(0,n)]
while len(leaves)<args.max_leaves:
    choices=[]
    for index,(start,end) in enumerate(leaves):
        old,_=fit(start,end)
        if end-start<2:continue
        options=[(fit(start,k)[0]+fit(k,end)[0],k) for k in range(start+1,end)]
        error,k=min(options)
        choices.append((old-error,index,k))
    if not choices:break
    gain,index,k=max(choices)
    if gain<1e-9:break
    start,end=leaves[index]
    leaves[index:index+1]=[(start,k),(k,end)]
    error=sum(fit(s,e)[0] for s,e in leaves)/(len(target)*18)
    print(json.dumps(dict(leaves=len(leaves),training_mse=error)),flush=True)
    if error<1e-12:break
thresholds=[float((times[end-1]+times[end])/2) for _,end in leaves[:-1]]
checkpoint=dict(schema='learned-pwa-actions-v1',action_family='absolute-joint-limits-v1',
    features=features,feature_transform=transform,thresholds=torch.tensor(thresholds),mean=torch.tensor(mean),
    scale=torch.tensor(scale),weights=torch.tensor(np.stack([fit(s,e)[1] for s,e in leaves])),
    dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
    training='greedy time partitions and least squares; disjoint episode validation',
    curriculum=True,formal_acceptance=False)
policy=PiecewiseAffinePolicy(checkpoint)
with torch.inference_mode():
    pred=policy(observations.reshape(-1,794)).reshape_as(actions)
    errors=(pred-actions).square()
    metrics=dict(kind='CPU supervised fit, not a simulation result',leaves=len(leaves),
        thresholds_s=thresholds,training_mse=float(errors[:,~hold].mean()),
        validation_mse=float(errors[:,hold].mean()),
        validation_max_abs_error=float((pred[:,hold]-actions[:,hold]).abs().max()),
        features=features,feature_transform=transform,global_training_window_s=[args.start_time,args.end_time],
        teacher_prefix_steps=0)
torch.save(checkpoint,args.output/'policy.pt')
(args.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics),flush=True)
