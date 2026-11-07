"""Measured-state MPC warm start from a validated persistent feedback policy.

The feedback seed is not automatically executable: the original nonlinear path
and terminal checks remain mandatory. Failed seeds fall back to V2 restoration.
This phase-conditioned diagnostic is not a general risk-triggered teacher.
"""
import copy
import time
from pathlib import Path
import numpy as np
from scripts.frictional_receding_recovery_v2 import FrictionalRecedingRecoveryV2
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts import search_frictional_recovery_cem_v1 as core
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance


class FrictionalFeedbackSeedRecoveryV3(FrictionalRecedingRecoveryV2):
    def __init__(self,configuration,spheres,feedback_policy,**kwargs):
        super().__init__(configuration,spheres,**kwargs)
        self.feedback_policy=feedback_policy
        self.feedback_spheres=spheres
        self.controller.sequence_geometry=AuthoredGroundClearance(
            Path(__file__).resolve().parents[1]/'configs/go2_piper_authored_ground_geometry_v1.json',
            self.controller.qp.model,margin_m=0.)

    def solve(self,observed_state,student_reference,warm_commands,*,control_tick):
        if type(control_tick) is not int or not 460<=control_tick<800:
            raise ValueError('Original observed episode phase required')
        started=time.monotonic();state=copy.deepcopy(observed_state);commands=[];failure=None
        geometry=self.controller.sequence_geometry
        for step in range(80):
            # Beyond the episode boundary the terminal controller is stationary
            # in phase; this extends only the planning horizon, not the task.
            offset=min(control_tick+step-460,339)
            target,_=self.feedback_policy.action(offset,state,state.previous_command_target)
            try:
                item=predict_frictional_spheres(self.controller.qp.model,self.controller.qp.interface,
                                                state,target,self.feedback_spheres,geometry=geometry)
            except (ValueError,np.linalg.LinAlgError) as error:
                failure=dict(step=step,error=repr(error));break
            d=item['diagnostics']
            bad={key:value for key,value in d.items() if (key=='nonfoot_ground_violation_m' and value>1e-6)
                 or (key in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm']
                     and value>3e-4)}
            if bad:failure=dict(step=step,violations=bad);break
            state=item['state'];state.previous_command_target=target.copy();commands.append(target.copy())
        verification=None
        if failure is None:
            seed=np.asarray(commands)
            core.CONTEXT=(self.controller,copy.deepcopy(observed_state),
                          observed_state.previous_command_target.copy(),seed,self.feedback_spheres,geometry)
            verification=core.evaluate(np.zeros_like(seed),save_states=False)
        accepted=verification is not None and verification['accepted']
        chosen=np.asarray(commands) if accepted else warm_commands
        old_limit=self.time_limit_s
        self.time_limit_s=max(1.,old_limit-(time.monotonic()-started))
        try:decision=super().solve(observed_state,student_reference,chosen)
        finally:self.time_limit_s=old_limit
        decision['feedback_seed']=dict(control_tick=control_tick,generated_steps=len(commands),
            generation_failure=failure,original_nonlinear_verification=verification,
            admitted_as_warm_start=bool(accepted),original_shifted_seed_used=not accepted)
        decision['planner_version']='persistent_feedback_warm_start_v3'
        decision['wall_seconds']=time.monotonic()-started
        return decision
