"""Sequential recovery rollout using interval-constrained position commands."""
import numpy as np
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.interval_command_support_qp_v1 import IntervalCommandSupportQP
from scripts.implicit_support_prediction_v1 import predict_implicit_support


class IntervalPredictionRejected(ValueError):
    pass


class LandingIntervalCommandMPCV1(LandingCommandHistoryMPCV1):
    qp_type=IntervalCommandSupportQP

    def __init__(self,interface,*,arm_target_rate_rad_s=5.,substeps=16,**kwargs):
        super().__init__(interface,arm_target_rate_rad_s=arm_target_rate_rad_s,**kwargs)
        old=self.qp
        self.qp=self.qp_type(interface,old.settings,arm_target_rate_rad_s=arm_target_rate_rad_s,
            substeps=substeps,minimum_root_height_m=.12,continuous_reference=old.continuous_reference,
            preserve_arm_command=old.preserve_arm_command,reference_frequency_rad_s=old.reference_frequency_rad_s,
            load_aware=old.load_aware,load_force_floor_n=old.load_force_floor_n)
        self.prediction_checks=[]

    def _integrate(self,state,acceleration,plan=None):
        if plan is None:raise IntervalPredictionRejected('Actual command required')
        target=self.admit_command_target(state,plan['joint_position_target'])
        result=predict_implicit_support(self.qp.model,self.qp.interface,state,target,
            duration_s=self.settings.control_dt_s,substeps=self.qp.substeps,
            contact_velocity_decay_s=self.qp.settings.contact_velocity_decay_s)
        diagnostics=result['diagnostics'].copy();following=result['state']
        diagnostics['qp_prediction_acceleration_disagreement']=float(np.max(abs(
            (following.v-state.v)/self.settings.control_dt_s-np.asarray(acceleration))))
        diagnostics['root_height_violation_m']=max(0.,self.qp.minimum_root_height_m-following.q[2])
        self.prediction_checks.append(diagnostics)
        physical=[value for key,value in diagnostics.items() if key not in
            ('maximum_active_set_iterations','qp_prediction_acceleration_disagreement')]
        if max(physical)>self.qp.settings.residual_tolerance or following.q[2]<self.qp.minimum_root_height_m-1e-10:
            raise IntervalPredictionRejected('Admitted interval prediction violates original physical constraints')
        if diagnostics['qp_prediction_acceleration_disagreement']>1e-2:
            raise IntervalPredictionRejected('Interval QP and admitted-command predictor disagree')
        following.previous_command_target=target.copy()
        return following

    def risk(self,state):
        result=super().risk(state)
        result['root_height_m']=float(state.q[2])
        if state.q[2]<.12:
            result.update(trigger=True,reason='original_task_root_height')
        return result

    def _terminal_ok(self,state,risk):
        return bool(state.q[2]>=.12 and super()._terminal_ok(state,risk))

    def solve(self,state,joint_reference,**kwargs):
        self.prediction_checks=[]
        try:result=super().solve(state,joint_reference,**kwargs)
        except IntervalPredictionRejected as error:
            result=dict(accepted=False,intervened=True,horizon=[],rejection=str(error),
                constraint_failure=True,solver_failure=False,risk=self.risk(state),
                native_execution=False,mpc_contract=self.contract())
        result['implicit_prediction_checks']=list(self.prediction_checks)
        return result

    def contract(self):
        result=super().contract()
        result.update(schema='landing-interval-command-mpc-v1',substeps=self.qp.substeps,
            formulation='sequential-QP-with-all-substep-command-constraints',minimum_root_height_m=.12,
            drive_branch='unsaturated-entire-interval',nonlinear_contact_and_impact_model_complete=False,
            native_recovery_validated=False)
        return result
