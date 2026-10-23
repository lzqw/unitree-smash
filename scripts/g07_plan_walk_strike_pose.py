#!/usr/bin/env python3
"""Conditional CPU arm seeds for the longer canonical walking setup."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.control.piper_geometry import PiperGeometry

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
old=ROOT/'reports/G07/eval_piecewise_curriculum01'
meta=json.loads((old/'metrics.json').read_text())
lane=meta['selected_video_lane']
data=torch.load(old/'policy_dataset.pt',map_location='cpu',weights_only=False)
q0=data['observations'][148,lane,21:27].numpy()+np.array([0.,1.,-1.,0.,0.,0.])
poses=np.load(old/'link_poses.npy')
names=json.loads((old/'body_names.json').read_text())
body=names.index('base')
pose=poses[74,body]
rpy=Rotation.from_quat(pose[[4,5,6,3]]).as_euler('xyz')
geometry=PiperGeometry()
target=np.array([1.45,0.,1.65])
normal=np.array([.917885687,0.,.396844889])
low=np.r_[geometry.bounds[:,0],[-.095,-.095]]
high=np.r_[geometry.bounds[:,1],[.095,.095]]
rng=np.random.default_rng(715201)
results=[]
for height in [.695,.735,.765]:
    base=np.array([float(pose[0])+.03,float(pose[1]),height])
    def residual(x):
        point,n=geometry.contact_center(x[:6],x[6:],base_position=base,base_rpy=rpy)
        return np.r_[20*(point-target),2*(n-normal),.002*(x[:6]-q0)]
    best=None
    for i in range(12):
        seed=np.r_[q0+[0.,0.,0.,0.,0.,0.] if i==0 else q0+rng.normal(0,.25,6),[.08,.08]]
        fit=least_squares(residual,np.clip(seed,low+1e-7,high-1e-7),bounds=(low,high),max_nfev=300)
        point,n=geometry.contact_center(fit.x[:6],fit.x[6:],base_position=base,base_rpy=rpy)
        result=dict(arm_q_rad=fit.x[:6].tolist(),uv_m=fit.x[6:].tolist(),
            base_position_m=base.tolist(),base_rpy=rpy.tolist(),ball_center_m=point.tolist(),
            face_normal=n.tolist(),position_error_m=float(np.linalg.norm(point-target)),
            normal_error=float(np.linalg.norm(n-normal)),conditional_root_height_m=height,
            status='conditional CPU IK search seed; no collision, dynamics or tracking guarantee')
        if best is None or result['position_error_m']+.1*result['normal_error']<best['position_error_m']+.1*best['normal_error']:best=result
    results.append(best)
(a.output/'metrics.json').write_text(json.dumps(dict(kind='CPU conditional IK only',target=target.tolist(),normal=normal.tolist(),results=results),indent=2)+'\n')
torch.save(dict(arm_pose=torch.tensor(results[0]['arm_q_rad'],dtype=torch.float32),result=results[0]),a.output/'pose_seed.pt')
print(json.dumps(results),flush=True)
