"""Signed per-node arm tracking constraints for full-prefix local models."""
import numpy as np

ARM_TRACKING_LIMIT_RAD=.25


def arm_tracking_response(result,arm_q_indices,expected_steps):
    trace=result.get('trajectory',[]);indices=np.asarray(arm_q_indices)
    if (result.get('path_valid') is not True or len(trace)!=expected_steps or expected_steps<1
            or indices.shape!=(6,) or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices<0) or len(np.unique(indices))!=6):
        raise ValueError('Complete verified prefix and six-joint mapping required')
    values=[]
    for k,row in enumerate(trace):
        q=np.asarray(row.get('q'),float);reference=np.asarray(row.get('planned_arm_reference_rad'),float)
        if (row.get('node')!=k or q.ndim!=1 or len(q)<=indices.max() or reference.shape!=(6,)
                or not np.isfinite(q).all() or not np.isfinite(reference).all()):
            raise ValueError('Finite ordered state/reference history required')
        signed=q[indices]-reference;recorded=float(row.get('arm_tracking_error_rad',np.nan))
        if not np.isfinite(recorded) or abs(np.max(abs(signed))-recorded)>1e-9:
            raise ValueError('Recorded arm tracking disagrees with actual state/reference')
        if np.max(abs(signed))>ARM_TRACKING_LIMIT_RAD:
            raise ValueError('Verified label contradicts original tracking limit')
        values.extend(signed)
    vector=np.asarray(values,float)
    return vector,np.full_like(vector,-ARM_TRACKING_LIMIT_RAD),np.full_like(vector,ARM_TRACKING_LIMIT_RAD)
