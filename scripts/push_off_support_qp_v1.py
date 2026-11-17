"""Offline push-off task adapter over the existing loaded fullbody support QP.

Desired velocity changes are objectives, never injected states or forces.
Flight and Recovery remain owned by their existing controllers. QP acceptance
is only an unsaturated backward-Euler proposal; complete nonlinear prefix validation is
mandatory. This does not alter any Recovery QP or default standing task.
"""
import numpy as np
import pinocchio as pin
from scripts.implicit_command_support_qp_v1 import ImplicitCommandSupportQP
from scripts.full_authority_position_interface_v1 import FullAuthorityPositionPDInterfaceV1
from scripts.whole_body_support_qp_v1 import SupportSettings


class PushOffSupportQPV1(ImplicitCommandSupportQP):
    def __init__(self,interface,settings=SupportSettings()):
        super().__init__(interface,settings,preserve_arm_command=True,continuous_reference=False)
        self.interface=FullAuthorityPositionPDInterfaceV1(interface)
        self._task=None
        self._momentum_rate=None

    def angular_task(self,state,snapshot,reference):
        if self._momentum_rate is None:
            return super().angular_task(state,snapshot,reference)
        # Hdot = Ag*a + dAg*v, about the moving COM in world axes.
        # All 24 DOFs contribute, including the prescribed moving arm.
        data=self.model.model.createData()
        pin.computeCentroidalMapTimeVariation(self.model.model,data,state.q,state.v)
        matrix=data.Ag[3:].copy();bias=data.dAg[3:]@state.v
        inertia=snapshot['centroidal_inertia_world']
        # Normalize to rad/s^2 using the current locked whole-body inertia.
        # This is only objective scaling, not a substituted dynamics model.
        return np.linalg.solve(inertia,matrix),np.linalg.solve(inertia,self._momentum_rate-bias)

    def reference(self,state,snapshot,joint_reference):
        if self._task is None:raise ValueError('Explicit causal baseline task required')
        baseline,com_accel,delta_com,delta_angular=self._task
        # No standing/upright servo is added during push-off. Preserve the
        # frozen reference's predicted angular and joint accelerations.
        return dict(com_position=snapshot['com_world'].copy(),
            com_acceleration=com_accel+delta_com,
            angular_acceleration_body=baseline[3:6]+delta_angular,
            joint_acceleration=baseline[self.model.v_indices].copy(),
            swing_foot_target_world=None)

    def propose(self,state,final_reference,baseline_acceleration,baseline_com_acceleration,
                delta_com_acceleration,delta_angular_acceleration,*,phase,recovery_owned=False,
                desired_angular_momentum_rate_world=None):
        reference=np.asarray(final_reference,float)
        if reference.shape!=(18,) or not np.isfinite(reference).all():raise ValueError('Finite final18 reference required')
        if np.any(reference<self.interface.limits[:,0]) or np.any(reference>self.interface.limits[:,1]):raise ValueError('Original command limits required')
        if recovery_owned or phase!='PUSH_OFF' or not state.contacts.any():
            return dict(accepted=False,owner='existing_controller',rejection='phase_authority',
                joint_position_target=reference.copy(),force_world=np.zeros((4,3)),native_execution=False)
        values=[np.asarray(x,float).copy() for x in (baseline_acceleration,baseline_com_acceleration,delta_com_acceleration,delta_angular_acceleration)]
        if any(x.shape!=shape or not np.isfinite(x).all() for x,shape in zip(values,((24,),(3,),(3,),(3,)))):
            raise ValueError('Finite causal acceleration tasks required')
        self.interface.bind_previous_command(state.previous_command_target)
        self._task=values
        if desired_angular_momentum_rate_world is not None:
            rate=np.asarray(desired_angular_momentum_rate_world,float)
            if rate.shape!=(3,) or not np.isfinite(rate).all():
                self._task=None
                raise ValueError('Finite whole-body angular momentum rate required')
            self._momentum_rate=rate.copy()
        try:plan=super().solve(state,reference)
        finally:self._task=None;self._momentum_rate=None
        if plan['accepted'] and np.max(abs(plan['joint_position_target'][12:]-reference[12:]))>2e-6:
            raise ValueError('Push-off replaced the prescribed arm command')
        plan.update(owner='push_off_qp_proposal',requested_com_acceleration_delta=values[2],
            requested_angular_acceleration_delta=values[3],fullbody_prediction_valid=False,
            native_execution=False,valid_teacher_reference=False,
            contact_model='existing implicit-drive measured foot-body QP; independent sphere validation required')
        plan['desired_angular_momentum_rate_world']=desired_angular_momentum_rate_world
        return plan
