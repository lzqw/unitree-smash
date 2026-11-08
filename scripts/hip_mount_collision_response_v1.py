"""Unilateral generalized reaction at audited hip/mount collision boundaries.

For the fixed base mount and one-axis hips, the nonintersection constraint is
an angular half-space. Its impulse is internal, solved together with ground
friction and the unchanged actuator through the full mass matrix. This is an
approximate zero-restitution collision response, not a new hardware limit or
a complete self-collision model. No native asset or command range is changed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.frictional_position_limited_step_v1 import position_limited_velocity_step

ROOT=Path(__file__).resolve().parents[1]


class HipMountCollisionResponseV1:
    def __init__(self,model,*,penetration_decay_s=.01):
        cfg=json.loads((ROOT/'configs/go2_piper_hip_mount_boundaries_v1.json').read_text())
        for path,digest in cfg['geometry_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed collision geometry')
        if not np.isfinite(penetration_decay_s) or penetration_decay_s<=0:raise ValueError('Explicit penetration decay required')
        self.decay=float(penetration_decay_s);self.constraints=[]
        for row in cfg['angular_boundaries']:
            j=model.joint_order.index(row['joint'])
            for b in row['authored_intersection_boundaries']:
                self.constraints.append((j,float(b['angle_rad']),bool(b['overlap_on_lower_side'])))
        if len(self.constraints)!=2:raise ValueError('Require the two audited hip/mount boundaries')

    def interval_bounds(self,position,physical_bounds,dt):
        q=np.asarray(position,float);bounds=np.asarray(physical_bounds,float).copy()
        if q.shape!=(18,) or bounds.shape!=(18,2) or not np.isfinite(dt) or dt<=0:
            raise ValueError('Original scalar joint state and interval required')
        for joint,bound,lower in self.constraints:
            # Correct existing tiny penetration gradually, using the same
            # declared time scale as ground penetration in the predictor.
            penetrated=q[joint]<bound if lower else q[joint]>bound
            effective=q[joint]+min(1.,dt/self.decay)*(bound-q[joint]) if penetrated else bound
            if lower:bounds[joint,0]=max(bounds[joint,0],effective)
            else:bounds[joint,1]=min(bounds[joint,1],effective)
        return bounds

    def step(self,*args,**kwargs):
        values=list(args);dt=values[8];q=values[13];physical=values[14]
        values[14]=self.interval_bounds(q,physical,dt)
        result=position_limited_velocity_step(*values,**kwargs)
        reaction=result['position_limit_impulse'].copy()
        mask=np.zeros(18,bool)
        for joint,bound,lower in self.constraints:
            mask[joint]=(reaction[joint]>0 if lower else reaction[joint]<0)
        result['hip_mount_impulse']=np.where(mask,reaction,0.)
        return result
