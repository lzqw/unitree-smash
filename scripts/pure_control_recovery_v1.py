"""Event-phase full-command predictive WBC candidate, without learned recovery.

The execution solver optimizes one physical control interval. It is not a
multi-step MPC. State-dependent foot placement and support goals are recomputed
at every call. Solver failure uses an explicitly uncertified bounded posture
candidate, never adds the frozen student's residual after ownership transfer.
"""
from pathlib import Path
import time
import numpy as np
import pinocchio as pin

from scripts.whole_body_support_qp_v1 import SupportSettings
from scripts.pure_control_interval_qp_v1 import PureControlIntervalQPBaseV1
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP

ROOT=Path(__file__).resolve().parents[1]


class PureControlRecoveryV1(PureControlIntervalQPBaseV1):
    def __init__(self,interface,nominal_posture):
        settings=SupportSettings(com_kp_s2=35.,com_kd_s=12.,orientation_kp_s2=35.,
            orientation_kd_s=12.,posture_kp_s2=25.,posture_kd_s=8.,
            swing_foot_weight=20.,solver_max_iterations=3000)
        super().__init__(interface,settings,continuous_reference=False,preserve_arm_command=False,
            arm_target_rate_rad_s=10.,substeps=16,minimum_root_height_m=0.)
        self.geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',
            self.model,margin_m=0.)
        self.nominal=np.asarray(nominal_posture,float).copy()
        if self.nominal.shape!=(18,) or np.any(self.nominal<interface.limits[:,0]) or np.any(self.nominal>interface.limits[:,1]):
            raise ValueError('Original in-range nominal posture required')
        self.started=None;self.initial_arm=None;self.initial_preload=None
        self.phase=None;self.elapsed=0.;self.reference_diagnostics={}

    def _make_solver(self):
        return DenseActiveSetQP()

    def reference(self,state,snapshot,joint_reference):
        r=state.root_rotation_world;yaw=np.arctan2(r[1,0],r[0,0]);c,s=np.cos(yaw),np.sin(yaw)
        upright=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        ids=[self.model.names.index(n) for n in self.model.feet]
        feet=snapshot['body_positions_world'][ids]
        qref=state.q.copy();qref[:3]=0.;qref[3:7]=pin.Quaternion(upright).coeffs()
        qref[self.model.q_indices]=joint_reference
        reference_feet=np.asarray(self.model.poses(qref)[0])[ids]
        reference_com=pin.centerOfMass(self.model.model,self.model.data,qref).copy()
        support=state.contacts
        center=(feet[support].mean(0) if support.any() else snapshot['com_world'].copy())
        # Ground is the actual unchanged horizontal plane. Foot-frame height
        # uses the loaded nominal footprint, translated to current support.
        height=float(reference_com[2]-reference_feet[:,2].mean())
        reference_center=reference_feet.mean(0)
        desired=snapshot['com_world'].copy()
        if support.any():
            desired[:2]=center[:2]
            desired[2]=center[2]+height-(.04 if self.phase=='BUFFER_LOAD' else 0.)
        cfg=self.settings
        acceleration=cfg.com_kp_s2*(desired-snapshot['com_world'])-cfg.com_kd_s*snapshot['com_velocity_world']
        if not support.any():acceleration=np.array([0.,0.,-9.81])
        swing=center+reference_feet-reference_center
        # Raibert/capture-style placement is a soft reference, with a fresh
        # measured velocity each tick; it is not a certified VHSIP solution.
        capture=np.sqrt(max(height,.1)/9.81)*snapshot['com_velocity_world'][:2]
        capture*=min(1.,.25/max(np.linalg.norm(capture),1e-12))
        swing[~support,:2]+=capture
        if support.any():swing[~support,2]=np.median(feet[support,2])
        else:
            # Preserve reachable nominal leg extension in flight, then
            # lower the world footprint as the body descends.
            swing[:,2]=state.q[2]+reference_feet[:,2]
        swing[support]=feet[support]
        angular=cfg.orientation_kp_s2*pin.log3(r.T@upright)-cfg.orientation_kd_s*state.v[3:6]
        self.reference_diagnostics=dict(com_goal=desired.copy(),swing_feet=swing.copy(),
            phase=self.phase,capture_offset=capture.copy(),support=support.copy())
        return dict(com_position=desired,com_acceleration=acceleration,
            angular_acceleration_body=angular,
            joint_acceleration=cfg.posture_kp_s2*(joint_reference-state.joint_position)-cfg.posture_kd_s*state.joint_velocity,
            swing_foot_target_world=swing)

    def command(self,state,phase,time_s):
        began=time.perf_counter()
        if phase not in ['AIR_PREPARE','BUFFER_LOAD','SUSTAIN']:raise ValueError('Recovery ownership required')
        if self.started is None:
            self.started=float(time_s);self.initial_arm=state.joint_position[12:].copy()
            self.initial_preload=state.previous_command_target-state.joint_position
        self.phase=phase;self.elapsed=float(time_s)-self.started
        if self.elapsed<0:raise ValueError('Reset controller on episode changes')
        reference=self.nominal.copy()
        # Arm braking has an evolving posture goal; the solver remains free to
        # choose all six arm commands jointly with the legs at every tick.
        decay=np.exp(-self.elapsed/.5)
        reference[12:]=self.nominal[12:]+decay*(self.initial_arm-self.nominal[12:])
        self.interface.bind_previous_command(state.previous_command_target)
        low,high=self.interface.admitted_target_bounds(state.joint_position,.01)
        reference=np.clip(reference,low,high)
        plan=self.solve(state,reference)
        if plan['accepted']:
            target=plan['joint_position_target'].copy();source='interval_qp'
        else:
            # Explicit experimental fallback. Preserve some actual actuator
            # preload while continuously following the current posture and
            # velocity; it is not certified by the rejected QP.
            target=reference.copy()
            target[:12]+=.5*np.exp(-self.elapsed/.4)*self.initial_preload[:12]
            target-=.02*state.joint_velocity
            target=np.clip(target,low,high);source='uncertified_posture_fallback'
        surfaces=authored_clearance_state(self.geometry,state)
        diagnostic=dict(phase=phase,solver_accepted=bool(plan['accepted']),source=source,
            solver_status=plan.get('solver_status'),rejection=plan.get('rejection'),
            residuals=plan.get('residuals'),minimum_authored_nonfoot_gap_m=plan.get('minimum_authored_nonfoot_gap_m'),
            head_lower=surfaces['Head_lower'],head_upper=surfaces['Head_upper'],
            reference=self.reference_diagnostics,wall_seconds=time.perf_counter()-began,
            sequence_mpc=False,native_recovery_certified=False,valid_expert_labels=False)
        return target,diagnostic
