"""Prefer small command changes only when the full predicted horizon passes.

If the command-cost rollout fails, replay the unmodified objective from the
same state, control tick and per-lane reference memory. That baseline result is
returned unchanged, including failure. No new first-step authority is granted.
"""
import time

from scripts.landing_minimum_command_mpc_v1 import LandingMinimumCommandMPCV1


class LandingFeasibleCommandCostMPCV1(LandingMinimumCommandMPCV1):
    def solve(self,state,joint_reference,**kwargs):
        started=time.perf_counter()
        kwargs=dict(kwargs)
        lane=int(kwargs.get('lane',0))
        if kwargs.get('control_tick') is None:
            kwargs['control_tick']=self._observed_tick.get(lane,-1)+1
        weight=self.qp.command_change_weight
        candidate=super().solve(state,joint_reference,**kwargs)
        fallback=None
        if weight>0 and not candidate['accepted'] and candidate.get('intervened'):
            self.qp.command_change_weight=0.
            try:
                fallback=super().solve(state,joint_reference,**kwargs)
            finally:
                self.qp.command_change_weight=weight
        elapsed=time.perf_counter()-started
        candidate_runtime=float(candidate.get('wall_runtime_s',0.))
        fallback_runtime=None if fallback is None else float(fallback.get('wall_runtime_s',0.))
        result=candidate if fallback is None else fallback
        result['command_cost_selection']=dict(
            candidate_runtime_s=candidate_runtime,baseline_runtime_s=fallback_runtime,
            total_runtime_s=elapsed,
            declared_weight=weight,selected_weight=weight if fallback is None else 0.,
            command_cost_horizon_accepted=bool(candidate['accepted']),
            baseline_fallback_evaluated=fallback is not None,
            baseline_horizon_accepted=None if fallback is None else bool(fallback['accepted']),
            selected='command_cost' if fallback is None else 'baseline',
            same_control_tick=kwargs['control_tick'],
            future_student_policy_predicted=False,
            full_horizon_acceptance_is_native_success=False)
        result['wall_runtime_s']=elapsed
        result['mpc_contract']=self.contract()
        return result

    def contract(self):
        result=super().contract()
        result.update(schema='landing-feasible-command-cost-v1',
            command_cost_selection='accept full-horizon candidate; otherwise unchanged-objective fallback',
            baseline_failure_not_upgraded=True,reference_advanced_once_per_actual_tick=True,
            native_recovery_validated=False)
        return result
