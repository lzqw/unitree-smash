"""Native development trial of one complete feedback reference on 26 failures.

Known case IDs select this offline experiment. This is not a deployable risk
trigger, a newly solved MPC, or automatically accepted expert data. Each
controller starts from its own actually reached state and command history.
"""
import copy
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

from scripts.native_timed_experimental_recovery_v1 import NativeTimedExperimentalRecoveryV1
from scripts.search_preserved_early_support_v1 import joint_factory
from scripts.timed_whole_body_recovery_v1 import build_timed_plan


CASES=(1,3,8,9,12,18,20,30,38,42,50,54,57,58,61,67,70,71,77,85,87,90,95,96,98,115)


class NativeTimedRecoveryFamilyV1:
    def __init__(self,bundle,case,limits,policy_sha256,output):
        self.inner=NativeTimedExperimentalRecoveryV1(bundle,case,limits,policy_sha256,output)
        self.env=None;self.histories={};self.policies={};self.family_records=[];self.alignments={}
        with np.load(self.inner.bundle/self.inner.manifest['bundle_name'],allow_pickle=False) as z:
            self.reference_q=z['tracking_reference_q'].copy();self.reference_v=z['tracking_reference_v'].copy()
        self.source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def __getattr__(self,name):
        return getattr(self.inner,name)

    def set_environment(self,env):
        if self.env is not None or env.num_envs!=128:raise ValueError('Single original environment required')
        self.env=env

    def _policy(self,case,state):
        # Rotate/translate the world reference in the horizontal plane only.
        # Ground height, joint knots, gains and physical settings are unchanged.
        original=self.inner.initial
        yaw=lambda r:float(np.arctan2(r[1,0],r[0,0]))
        angle=yaw(state.root_rotation_world)-yaw(original.root_rotation_world)
        rotation=pin.AngleAxis(angle,np.array([0.,0.,1.])).matrix()
        rq=self.reference_q.copy()
        shift=state.q[:3]-rotation@original.q[:3];shift[2]=0.
        rq[:,:3]=rq[:,:3]@rotation.T+shift
        for row in rq:row[3:7]=pin.Quaternion(rotation@pin.Quaternion(row[3:7]).matrix()).coeffs()
        worker=dict(controller=self.inner.controller,initial=copy.deepcopy(state),
            factory=partial(joint_factory,reference_q=rq,reference_v=self.reference_v,
                            gains=self.inner.manifest['tracking_gains']))
        self.alignments[str(case)]=dict(yaw_rotation_rad=angle,translation_world_m=shift.tolist(),
            initial_q=state.q.tolist(),initial_v=state.v.tolist(),previous_target_rad=state.previous_command_target.tolist())
        return build_timed_plan(worker,self.inner.manifest['proposal'])

    def action(self,tick,public,student,finished,native_state=None):
        if self.env is None:raise ValueError('Actual environment required')
        output=self.inner.action(tick,public,student,finished,native_state)
        for case in CASES:
            if case==85:continue
            if tick<280:
                self.histories[case]=self.inner.decode(student[case].detach().cpu().numpy())
                continue
            if bool(finished[case]):continue
            if abs(float(public[case,775])-.01*tick)>1e-5:raise ValueError('Original causal clock required')
            state=self.model.state_from_native(self.env,case)
            previous=self.histories[case].copy();state.previous_command_target=previous
            if tick==280:self.policies[case]=self._policy(case,state)
            target,mode=self.policies[case].action(80+tick-280,state,previous.copy())
            target=np.asarray(target,float)
            if (target.shape!=(18,) or not np.isfinite(target).all()
                    or np.any(target<self.limits[:,0]-1e-7) or np.any(target>self.limits[:,1]+1e-7)
                    or np.max(abs(target[12:]-previous[12:]))>.100001):
                raise ValueError('Original interface violation')
            output[case]=torch.as_tensor(2*(target-self.limits[:,0])/np.diff(self.limits).ravel()-1,
                                         dtype=student.dtype,device=student.device)
            realized=self.inner.decode(output[case].detach().cpu().numpy())
            if np.max(abs(realized[12:]-previous[12:]))>.100001:raise ValueError('Encoded slew violation')
            self.histories[case]=realized
            self.family_records.append(dict(case=case,tick=tick,mode=mode,q=state.q.copy(),v=state.v.copy(),
                contacts=state.contacts.copy(),foot_force_world_n=state.measured_force_world.copy(),
                previous_target_rad=previous,requested_target_rad=target.copy(),final_target_rad=realized.copy(),
                student_action=student[case].detach().cpu().numpy().copy(),
                final_action=output[case].detach().cpu().numpy().copy()))
        return output

    def save(self):
        result=self.inner.save()
        fields=['case','tick','q','v','contacts','foot_force_world_n','previous_target_rad',
                'requested_target_rad','final_target_rad','student_action','final_action']
        np.savez_compressed(self.output/'recovery_family_trace.npz',
                            **{k:np.asarray([r[k] for r in self.family_records]) for k in fields})
        metadata=dict(scope=__doc__,cases=list(CASES),reference_case=85,start_tick=280,end_tick=800,
            records=len(self.family_records),alignments=self.alignments,source_sha256=self.source_hash,
            native_execution=True,online_mpc=False,valid_expert_labels=False,automatic_promotion=False,
            controls_selected_by_known_case_ids=True,controller_parameters_changed=False,
            physical_parameters_changed=False,task_scoring_changed=False)
        (self.output/'recovery_family_result.json').write_text(json.dumps(metadata,indent=2)+'\n')
        return result
