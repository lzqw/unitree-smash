"""Adapter from a complete TerminalPushRollout result to planner evidence."""
import numpy as np

class EventCoupledRolloutAdapterV1:
    def __init__(self, rollout, *, impact_result=None, finite_mass_status='EFFECTIVE_MASS_UNVERIFIED'):
        self.rollout=rollout
        self.impact_result=impact_result
        self.finite_mass_status=finite_mass_status
    def __call__(self, theta):
        result=self.rollout(np.asarray(theta,float))
        if not isinstance(result,dict) or result.get('path_valid') is not True or not result.get('trajectory'):
            return dict(path_valid=False,finite_mass_status=self.finite_mass_status)
        terminal=result['terminal_errors']
        out=dict(path_valid=True,contact_position_error_m=float(terminal['position_m']),
                 normal_error=float(terminal['normal']),velocity_error_m_s=float(terminal['velocity_m_s']),
                 finite_mass_status=self.finite_mass_status,trajectory=result['trajectory'],
                 final_impact=result.get('final_impact'),native_reference_valid=False,teacher_eligible=False)
        if self.impact_result is not None: out['final_impact']=self.impact_result
        return out
