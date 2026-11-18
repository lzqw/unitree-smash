"""Adapter from a complete TerminalPushRollout result to planner evidence."""
import numpy as np

class EventCoupledRolloutAdapterV1:
    def __init__(self, rollout, *, impact_builder=None):
        self.rollout=rollout
        self.impact_builder=impact_builder
    def __call__(self, theta):
        result=self.rollout(np.asarray(theta,float))
        if not isinstance(result,dict) or result.get('path_valid') is not True or not result.get('trajectory'):
            return dict(path_valid=False,finite_mass_status='EFFECTIVE_MASS_UNVERIFIED')
        terminal=result['terminal_errors']
        out=dict(path_valid=True,contact_position_error_m=float(terminal['position_m']),
                 normal_error=float(terminal['normal']),velocity_error_m_s=float(terminal['velocity_m_s']),
                 finite_mass_status='EFFECTIVE_MASS_UNVERIFIED',trajectory=result['trajectory'],
                 native_reference_valid=False,teacher_eligible=False)
        if self.impact_builder is not None:
            evidence=self.impact_builder(result)
            if not isinstance(evidence,dict): raise ValueError('Impact builder must return evidence')
            forbidden={'path_valid','contact_position_error_m','normal_error','velocity_error_m_s',
                       'terminal_errors','trajectory','native_reference_valid','teacher_eligible'}
            if forbidden.intersection(evidence):
                raise ValueError('Impact evidence cannot overwrite rollout evidence')
            out.update(evidence)
        return out
