"""Read actual USD leg joints and compare nominal target posture geometry."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
from pxr import Usd, UsdPhysics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ASSET = ROOT/'assets/quadsmash_full_racket_v4_r3.usd'
stage = Usd.Stage.Open(str(ASSET))

def rotation(wxyz):
    w,x,y,z = np.array(wxyz)/np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def frame(position, q):
    t=np.eye(4);t[:3,:3]=rotation([q.GetReal(),*q.GetImaginary()]);t[:3,3]=position
    return t

joints = {}
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):continue
    j=UsdPhysics.Joint(prim)
    if prim.GetName().split('_')[0] not in ['FL','FR','RL','RR']:continue
    axis=prim.GetAttribute('physics:axis').Get()
    joints[prim.GetName()] = (frame(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get()),
                             frame(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get()),axis)

def feet(q):
    result=[]
    for i,leg in enumerate(['FL','FR','RL','RR']):
        t=np.eye(4)
        for k,name in enumerate(['hip','thigh','calf','foot']):
            a,b,axis=joints[f'{leg}_{name}_joint']; turn=np.eye(4)
            if axis is not None:
                angle=q[i+4*k]; vector=np.zeros(3);vector['XYZ'.index(axis)]=math.sin(angle/2)
                turn[:3,:3]=rotation([math.cos(angle/2),*vector])
            t=t@a@turn@np.linalg.inv(b)
        result.append(t[:3,3])
    return np.array(result)

snapshot=OUT.parent/'standing_recovery37_native_gen0_tick500_snapshot'
public=np.load(snapshot/'partial_public_and_poses.npz')['observations'][0,0]
poses=np.load(snapshot/'all_first_episodes.partial.npz')['link_poses'][0,0]
default=np.array([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5])
actual=feet(default+public[9:21])
native=(poses[18:22,:3]-poses[0,:3])@rotation(poses[0,3:])
error=float(np.max(np.linalg.norm(actual-native,axis=1)));assert error<5e-5,error
report=json.loads((OUT/'REPORT.json').read_text())
rows=[]
for thigh,calf in report['controlled_standing_thigh_calf_bias_pairs_rad']:
    q=default.copy();q[4:8]+=thigh;q[8:12]+=calf
    rows.append(dict(thigh_bias_rad=thigh,calf_bias_rad=calf,
                     feet_relative_to_root_m=feet(q).tolist(),
                     downward_foot_extent_m=(-feet(q)[:,2]).tolist()))
value=dict(created_at=datetime.now(timezone.utc).isoformat(),
           scope='Exact authored USD leg joint kinematics at nominal target postures, actor output/front-back perturbation zero. Not executed posture tracking or landing dynamics.',
           actual_initial_native_four_foot_reconstruction_error_m=error,
           rows=rows, hardware_and_joints_unchanged=True, native_execution_of_proposals=False,
           formal_acceptance=False,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ASSET,Path(__file__),snapshot/'partial_public_and_poses.npz',snapshot/'all_first_episodes.partial.npz']})
(OUT/'LEG_TARGET_GEOMETRY.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(native_reconstruction_error_m=error,nominal_downward_foot_extents_m=[r['downward_foot_extent_m'] for r in rows])))
