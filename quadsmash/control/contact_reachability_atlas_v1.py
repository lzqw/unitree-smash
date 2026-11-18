"""Read-only contact reachability atlas for causal candidate states."""
import numpy as np
from .contact_pose_reachability_v1 import diagnose_contact_pose

class ContactReachabilityAtlasV1:
    def __init__(self, *, kinematics_fn, lower, upper, speed, base_velocity_fn=None):
        self.kinematics_fn=kinematics_fn
        self.lower=np.asarray(lower,float);self.upper=np.asarray(upper,float);self.speed=np.asarray(speed,float)
        if self.lower.shape!=(6,) or self.upper.shape!=(6,) or self.speed.shape!=(6,): raise ValueError('Six-axis bounds required')
        if np.any(self.lower>=self.upper) or np.any(self.speed<=0): raise ValueError('Valid arm bounds required')
        self.base_velocity_fn=base_velocity_fn
    def evaluate(self,candidates):
        rows=[]
        for index,c in enumerate(candidates):
            required=('time_s','position_m','normal','velocity_m_s','q_seed_rad')
            if any(k not in c for k in required): raise ValueError('Incomplete atlas candidate')
            p=np.asarray(c['position_m'],float);n=np.asarray(c['normal'],float);v=np.asarray(c['velocity_m_s'],float);q=np.asarray(c['q_seed_rad'],float)
            if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,n,v)) or q.shape!=(6,) or not np.isfinite(q).all(): raise ValueError('Finite atlas state required')
            d=diagnose_contact_pose(self.kinematics_fn,q,self.lower,self.upper,self.speed,p,n,v,base_velocity_fn=self.base_velocity_fn)
            rows.append(dict(index=index,time_s=float(c['time_s']),source=c.get('source','unknown'),diagnosis=d,kinematic_only=True,fullbody_verified=False,native_verified=False,teacher_eligible=False))
        rows.sort(key=lambda x:min(s['position_error_m']/0.035+s['normal_vector_error']/0.12+s['velocity_error_m_s']/0.75 for s in x['diagnosis']['stages'].values()))
        return dict(candidates=rows,kinematic_only=True,global_infeasibility_proven=False,teacher_eligible=False)
