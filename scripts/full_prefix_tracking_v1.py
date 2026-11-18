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


def tracking_linearization_reserve(record,arm_q_indices):
    """Empirical local-model headroom, not a physical or probabilistic bound.

    Scale the last accepted discrepancy by inverse step-scale squared, a
    smooth-model truncation-error heuristic. Contact changes can invalidate it;
    complete nonlinear validation remains mandatory.
    """
    trial=record['trials'][record['selected_index']]
    if trial.get('path_response_valid') is not True:raise ValueError('Verified tracking response required')
    actual,_,_=arm_tracking_response(trial['result'],arm_q_indices,record['metadata']['contact_index'])
    logged=np.asarray(trial.get('actual_path_response'),float)
    predicted=np.asarray(trial.get('predicted_path_response'),float)
    scale=float(record.get('accepted_step_scale',np.nan))
    if (logged.shape!=actual.shape or not np.array_equal(logged,actual) or predicted.shape!=actual.shape
            or not np.isfinite(predicted).all() or not np.isfinite(scale) or not 0<scale<=1):
        raise ValueError('Matching finite predicted/actual responses and accepted scale required')
    error=float(np.max(abs(actual-predicted)));reserve=error/(scale*scale)
    if not np.isfinite(reserve):raise ValueError('Finite model reserve required')
    return reserve,dict(observed_max_error_rad=error,accepted_step_scale=scale,
        requested_reserve_rad=reserve,scaling='inverse accepted step scale squared',
        guaranteed_bound=False,physical_limit_rad=ARM_TRACKING_LIMIT_RAD,
        final_nonlinear_validation_required=True)
