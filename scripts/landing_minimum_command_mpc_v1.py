"""Whole-body interval QP with an explicit cost on student command changes.

The added term is .5*w*||u-u_student||^2 in radians, after the parent QP's
objective normalization. Original interval and joint-damper constraints remain.
The horizon uses the current requested command as a held reference; this is NOT
a prediction of future student-policy actions or a guarantee of intervention gain.
"""
import numpy as np
from scipy import sparse
from scripts.landing_joint_damper_mpc_v1 import JointDamperIntervalQP
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1


def command_change_cost(center, scale, student, weight):
    center,scale,student=[np.asarray(a,dtype=float) for a in (center,scale,student)]
    if (center.shape!=(18,) or scale.shape!=(18,) or student.shape!=(18,)
            or not all(np.isfinite(a).all() for a in (center,scale,student))
            or np.any(scale<=0) or not np.isfinite(weight) or weight<0):
        raise ValueError('Finite command reference, positive scales and nonnegative weight required')
    return weight*scale**2, weight*scale*(center-student)


class MinimumCommandIntervalQP(JointDamperIntervalQP):
    def __init__(self,*args,command_change_weight=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.isfinite(command_change_weight) or command_change_weight<0:
            raise ValueError('Finite nonnegative command-change weight required')
        self.command_change_weight=float(command_change_weight)
        self.student_command_reference=None
        self._tracking_context=None

    def _make_solver(self):
        solver=super()._make_solver()
        if self._tracking_context is None:
            raise RuntimeError('Student command context must precede QP construction')
        center,scale,student=self._tracking_context
        diagonal,linear=command_change_cost(center,scale,student,self.command_change_weight)
        original_setup=solver.setup
        # Only this fresh local solver instance is adapted; no shared solver
        # class, source file, constraint, or live controller is monkey-patched.
        def setup(**kwargs):
            kwargs['P']=kwargs['P']+sparse.diags(diagonal,format='csc')
            kwargs['q']=kwargs['q']+linear
            return original_setup(**kwargs)
        solver.setup=setup
        return solver

    def solve(self,state,joint_reference,*,static_equilibrium=False):
        interface=self.interface
        interface.bind_previous_command(state.previous_command_target)
        low,high=interface.admitted_target_bounds(state.joint_position,self.settings.control_dt_s)
        center=np.clip(np.asarray(joint_reference,dtype=float),low,high)
        original=(joint_reference if self.student_command_reference is None else
                  self.student_command_reference)
        student=np.clip(np.asarray(original,dtype=float),low,high)
        scale=.5*np.diff(interface.limits,axis=1).ravel()
        command_change_cost(center,scale,student,self.command_change_weight)
        self._tracking_context=(center,scale,student)
        try:
            result=super().solve(state,joint_reference,static_equilibrium=static_equilibrium)
        finally:
            self._tracking_context=None
        result.update(command_change_weight=self.command_change_weight,
            command_change_reference=student.tolist(),
            command_change_cost_scope='physical-radian command difference after parent objective normalization',
            student_reference_scope='held-current-request; no future-policy prediction',
            physical_constraints_relaxed=False)
        if 'joint_position_target' in result:
            delta=result['joint_position_target']-student
            result['student_command_change_l2_rad']=float(np.linalg.norm(delta))
            result['student_command_change_max_rad']=float(np.max(abs(delta)))
        return result


class LandingMinimumCommandMPCV1(LandingFullCommandIntervalMPCV1):
    qp_type=MinimumCommandIntervalQP

    def __init__(self,*args,command_change_weight=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.isfinite(command_change_weight) or command_change_weight<0:
            raise ValueError('Finite nonnegative command-change weight required')
        self.qp.command_change_weight=float(command_change_weight)

    def solve(self,state,joint_reference,**kwargs):
        requested=np.asarray(joint_reference,dtype=float)
        if requested.shape!=(18,) or not np.isfinite(requested).all():
            raise ValueError('Finite full student command required')
        self.qp.student_command_reference=requested.copy()
        try:
            result=super().solve(state,joint_reference,**kwargs)
        finally:
            self.qp.student_command_reference=None
        result['command_change_weight']=self.qp.command_change_weight
        result['student_command_reference_scope']='held-current-request; no future-policy prediction'
        return result

    def contract(self):
        result=super().contract()
        result.update(schema='landing-minimum-command-interval-v1',
            command_change_weight=self.qp.command_change_weight,
            command_change_cost='0.5*w*||actual_position_target-admitted_current_student_target||^2',
            student_future_policy_rolled_out=False,
            physical_constraints_relaxed=False,native_recovery_validated=False)
        return result
