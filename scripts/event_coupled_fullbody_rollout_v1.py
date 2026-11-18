"""Production joint 18D rollout over the existing full-body dynamics.

The parameter vector is [leg_position12, arm_position6, arm_rate6]. Every
candidate starts from the supplied actual state/history and is independently
rolled by CoupledContactRollout. No future state injection is possible.
"""
import numpy as np
from scripts.coupled_contact_prefix_v1 import CoupledContactRollout

class EventCoupledFullBodyRolloutV1:
    parameter_size=24
    def __init__(self, model, interface, predictor, state, commands, arm_reference, metadata):
        self.rollout=CoupledContactRollout(model,interface,predictor,state,commands,arm_reference,metadata)
    def __call__(self, theta):
        theta=np.asarray(theta,float)
        if theta.shape!=(self.parameter_size,) or not np.isfinite(theta).all(): raise ValueError('24 finite joint parameters required')
        result=self.rollout(theta)
        if not result.get('path_valid') or not result.get('trajectory'):
            return dict(path_valid=False,failure=result.get('failure'),finite_mass_status='EFFECTIVE_MASS_UNVERIFIED')
        e=result['terminal_errors']
        return dict(path_valid=True,contact_position_error_m=float(e['position_m']),normal_error=float(e['normal']),velocity_error_m_s=float(e['velocity_m_s']),trajectory=result['trajectory'],terminal_errors=e,finite_mass_status='EFFECTIVE_MASS_UNVERIFIED',native_reference_valid=False,teacher_eligible=False)
