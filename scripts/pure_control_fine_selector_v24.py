"""Use the planning time step when ranking executable feedback candidates.

The existing selector used 16 substeps against a 160-substep reference. Near
impact this can prefer a large correction to an exactly reproducible plan.
Keep the same candidates/objective/horizon/physical limits, but compare all
on the fine model. A zero-cost planned action is a proven lower bound of this
nonnegative local tracking objective, so remaining candidates can be pruned.
This is still restricted candidate control, not a global MPC certificate.
"""
import numpy as np
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.pure_control_state_settling_v23 import PureControlStateSettlingV23
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres


class FineRecoverySelectorMixin:
    def __init__(self, *args, **kwargs):
        self._zero_cost_planned_candidate = False
        super().__init__(*args, **kwargs)

    def _candidate(self, state, target, k):
        if self._zero_cost_planned_candidate:
            return dict(accepted=False, pruned_nonnegative_cost_lower_bound=True,
                model_rejected=False, native_certified=False)
        predicted = state
        cost = 0.
        horizon = min(2, len(self.chosen['trajectory'])-k)
        for offset in range(horizon):
            command = target if offset == 0 else np.asarray(self.chosen['trajectory'][k+offset]['target'])
            try:
                result = predict_self_contact_spheres(self.model,self.interface,predicted,command,
                    self.geometry.spheres,geometry=self.path_geometry,substeps=160)
            except (ValueError,np.linalg.LinAlgError) as error:
                return dict(accepted=False,error=repr(error),horizon_step=offset,substeps=160)
            bad = {n:v for n,v in result['diagnostics'].items() if
                (n=='nonfoot_ground_violation_m' and v>1e-6) or
                (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s',
                    'motor_torque_violation_nm'] and v>3e-4)}
            if bad:
                return dict(accepted=False,violations=bad,horizon_step=offset,substeps=160)
            predicted = result['state']
            predicted.previous_command_target = command.copy()
            cost += self._tracking_cost(predicted,self.library.state(self.chosen,k+offset))
        if cost == 0. and np.array_equal(target,np.asarray(self.chosen['trajectory'][k]['target'])):
            self._zero_cost_planned_candidate = True
        return dict(accepted=True,cost=cost,prediction_intervals=horizon,substeps=160,
            native_certified=False)

    def command(self, state, phase, time_s):
        self._zero_cost_planned_candidate = False
        target, diagnostic = super().command(state,phase,time_s)
        diagnostic.update(controller='pure_control_fine_selector_v24',
            feedback_candidate_substeps=160,
            zero_cost_planned_candidate=self._zero_cost_planned_candidate,
            global_sequence_optimum=False)
        return target, diagnostic


class PureControlFinePlanEndV24(FineRecoverySelectorMixin,PureControlStanceFootReferenceV15):
    pass


class PureControlFineStateSettlingV24(FineRecoverySelectorMixin,PureControlStateSettlingV23):
    pass
