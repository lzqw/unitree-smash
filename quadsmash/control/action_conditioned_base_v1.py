"""Immutable base transport from a causal fullbody command rollout.

This is an optimization reference, never a validity certificate. Changing any
18D command requires re-predicting and revalidating the complete trajectory.
No interpolation or extrapolation of an unverified future base is permitted.
"""
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation


def _array(value, shape, name):
    x=np.array(value,dtype=np.float64,copy=True)
    if x.shape!=shape or not np.isfinite(x).all():raise ValueError('Invalid '+name)
    return x


def _rotation(q):
    if abs(np.linalg.norm(q)-1)>1e-6:raise ValueError('Nonunit root quaternion')
    return Rotation.from_quat(q).as_matrix()


def context_digest(q,v,targets):
    h=hashlib.sha256()
    for value in (q,v,targets):
        x=np.asarray(value,dtype='<f8');h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


class ActionConditionedBaseV1:
    def __init__(self,initial_q,initial_v,targets,state_trace,*,dt_s=.01):
        if dt_s!=.01:raise ValueError('Original 10ms control period required')
        q0=_array(initial_q,(25,),'initial q');v0=_array(initial_v,(24,),'initial v')
        if not state_trace:raise ValueError('Complete predicted prefix required')
        count=len(state_trace)
        commands=_array(targets,(count,18),'final18 commands')
        qs=[q0];vs=[v0]
        for node,row in enumerate(state_trace):
            if row['node']!=node or abs(row['elapsed_s']-(node+1)*dt_s)>1e-9:
                raise ValueError('Nonsequential prediction trace')
            logged=_array(row['final_target_rad'],(18,),'logged final command')
            if not np.array_equal(logged,commands[node]):raise ValueError('Command provenance mismatch')
            qs.append(_array(row['q'],(25,),'predicted q'))
            vs.append(_array(row['v'],(24,),'predicted v'))
        rotations=np.array([_rotation(q[3:7]) for q in qs])
        basis=rotations[0]
        self._rotation=np.array([basis.T@r for r in rotations])
        self._translation=np.array([basis.T@(q[:3]-q0[:3]) for q in qs])
        self._linear=np.array([basis.T@r@v[:3] for r,v in zip(rotations,vs)])
        self._angular=np.array([basis.T@r@v[3:6] for r,v in zip(rotations,vs)])
        self._digest=context_digest(q0,v0,commands)
        self.dt_s=dt_s;self.horizon_s=count*dt_s
        for x in (self._rotation,self._translation,self._linear,self._angular):x.flags.writeable=False

    def validate_context(self,initial_q,initial_v,targets):
        if context_digest(initial_q,initial_v,targets)!=self._digest:
            raise ValueError('Stale base reference: initial state or final18 commands changed')

    def __call__(self,duration_s):
        t=float(duration_s)
        if not np.isfinite(t):raise ValueError('Finite event time required')
        node=round(t/self.dt_s)
        if node<0 or node>=len(self._translation) or abs(t-node*self.dt_s)>1e-8:
            raise ValueError('Requested event outside predicted control grid')
        return dict(rotation_current_from_future=self._rotation[node].copy(),
            translation_current_m=self._translation[node].copy(),
            linear_velocity_current_m_s=self._linear[node].copy(),
            angular_velocity_current_rad_s=self._angular[node].copy(),
            source='causal_action_conditioned_model',command_context_sha256=self._digest,
            execution_valid=False,requires_full_revalidation=True)
