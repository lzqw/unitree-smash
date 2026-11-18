"""Production joint 18D rollout over the existing full-body dynamics.

The parameter vector is [leg_position12, arm_position6, arm_rate6]. Every
candidate starts from the supplied actual state/history and is independently
rolled by CoupledContactRollout. No future state injection is possible.
"""
import numpy as np
from scripts.coupled_contact_prefix_v1 import CoupledContactRollout

class EventCoupledFullBodyRolloutV1:
    parameter_size=24
    def __init__(self, model, interface, predictor, state, commands, arm_reference, metadata, *, impact_builder=None):
        self.rollout=CoupledContactRollout(model,interface,predictor,state,commands,arm_reference,metadata)
        self.impact_builder=impact_builder
    def __call__(self, theta):
        theta=np.asarray(theta,float)
        if theta.shape!=(self.parameter_size,) or not np.isfinite(theta).all(): raise ValueError('24 finite joint parameters required')
        result=self.rollout(theta)
        if not result.get('path_valid') or not result.get('trajectory'):
            return dict(path_valid=False,failure=result.get('failure'),finite_mass_status='EFFECTIVE_MASS_UNVERIFIED')
        e=result['terminal_errors']
        out=dict(path_valid=True,contact_position_error_m=float(e['position_m']),normal_error=float(e['normal']),velocity_error_m_s=float(e['velocity_m_s']),trajectory=result['trajectory'],terminal_errors=e,native_reference_valid=False,teacher_eligible=False,contact_feasible=False,legal_return_predicted=False,attack5_predicted=False,target_landing_predicted=False)
        if self.impact_builder is None:
            out['finite_mass_status']='EFFECTIVE_MASS_UNVERIFIED'
        else:
            evidence=self.impact_builder(result['trajectory'][-1], theta)
            if not isinstance(evidence,dict): raise ValueError('Impact builder must return evidence')
            out.update(evidence)
            out.setdefault('finite_mass_status','EFFECTIVE_MASS_UNVERIFIED')
        return out
