"""Native evaluation of a declared, model-rejected recovery proposal.

This is a simulation diagnostic, not an admitted teacher or expert label.
No fabricated complete prediction is supplied to the verified-plan loader.
All commands use actual state and the original actuator interface.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch

from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.current_state_recovery_planner_v3 import build_feedback_plan
from scripts.search_preserved_early_support_v1 import joint_factory
from functools import partial


class NativeExperimentalRecoveryV2:
    def __init__(self,bundle,case,limits,policy_sha256,output):
        self.bundle=Path(bundle);self.output=Path(output)
        self.manifest=json.loads((self.bundle/'manifest.json').read_text());m=self.manifest
        if (m['contract']!='model_rejected_native_diagnostic_v1' or m['model_accepted'] or
                m['valid_expert_labels'] or m['policy_sha256']!=policy_sha256 or case!=m['case']):
            raise ValueError('Explicit model-rejected simulation diagnostic required')
        for path,digest in m['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed proposal input '+path)
        archive=self.bundle/m['bundle_name']
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256']:raise ValueError('Changed reference archive')
        self.case=int(case);self.start=280;self.steps=520;self.applied=0
        self.previous=None;self.last_tick=-1;self.records=[];self.query_check=None;self.early_termination_tick=None
        self.limits=np.asarray(limits.detach().cpu(),float)
        if not np.allclose(self.limits,m['configuration']['interface']['limits'],atol=1e-7,rtol=0):
            raise ValueError('Changed native limits')
        self.controller=build_controller(m['configuration']);self.model=self.controller.qp.model
        if m['configuration']['arm_target_rate_rad_s']!=10.:raise ValueError('Changed arm interface')
        # Native processes do not own the offline search worker globals.
        q=np.asarray(m['query']['q'],float).copy();v=np.asarray(m['query']['v'],float).copy()
        contacts=np.asarray(m['query']['contacts'],bool).copy()
        self.initial=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),
            q[self.model.q_indices].copy(),v[self.model.v_indices].copy(),contacts,0.)
        self.initial.previous_command_target=np.asarray(m['query']['target'],float).copy()
        with np.load(archive,allow_pickle=False) as z:
            rq=z['tracking_reference_q'].copy();rv=z['tracking_reference_v'].copy()
            self.query_public=z['query_public'].copy();self.query_previous_action=z['previous_action'].copy()
        worker=dict(controller=self.controller,initial=self.initial,
            factory=partial(joint_factory,reference_q=rq,reference_v=rv,gains=m['tracking_gains']))
        self.feedback=build_feedback_plan(worker,m['proposal'])

    def decode(self,action):
        return self.limits[:,0]+.5*(np.asarray(action)+1)*np.diff(self.limits).ravel()

    def action(self,tick,public,student,finished,native_state=None):
        if (public.shape!=(128,794) or student.shape!=(128,18) or finished.shape!=(128,)
                or tick!=self.last_tick+1):raise ValueError('Original128 continuous control required')
        self.last_tick=tick;output=student.clone();base=student[self.case].detach().cpu().numpy()
        if tick<self.start:self.previous=self.decode(base);return output
        if bool(finished[self.case]):
            if self.early_termination_tick is None:self.early_termination_tick=tick
            return output
        if native_state is None or self.previous is None:raise ValueError('Actual state/history required')
        raw=public[self.case].detach().cpu().numpy();offset=tick-self.start
        if not 0<=offset<520 or abs(float(raw[775])-.01*tick)>1e-5:raise ValueError('Original episode clock required')
        force=np.asarray(getattr(native_state,'measured_force_world',None))
        if force.shape!=(4,3) or not np.isfinite(force).all():raise ValueError('Native forces required')
        if offset==0:
            errors=dict(configuration_tangent=float(np.max(abs(pin.difference(self.model.model,self.initial.q,native_state.q)))),
                velocity=float(np.max(abs(native_state.v-self.initial.v))),
                public=float(np.max(abs(raw-self.query_public))),
                previous_target_rad=float(np.max(abs(self.previous-self.decode(self.query_previous_action)))))
            tolerances=dict(configuration_tangent=2e-5,velocity=2e-4,public=5e-4,previous_target_rad=5e-6)
            contact_equal=bool(np.array_equal(native_state.contacts,self.initial.contacts))
            self.query_check=dict(passed=contact_equal and all(errors[k]<=v for k,v in tolerances.items()),
                errors=errors,tolerances=tolerances,contacts_equal=contact_equal)
            (self.output/'sequence_query_match.json').write_text(json.dumps(self.query_check,indent=2)+'\n')
            if not self.query_check['passed']:raise ValueError('Original measured query differs; proposal not executed')
        state=copy.deepcopy(native_state);state.previous_command_target=self.previous.copy()
        target,mode=self.feedback.action(80+offset,state,self.previous.copy());target=np.asarray(target,float)
        if (target.shape!=(18,) or not np.isfinite(target).all() or
                np.any(target<self.limits[:,0]-1e-7) or np.any(target>self.limits[:,1]+1e-7) or
                np.max(abs(target[12:]-self.previous[12:]))>.100001):
            raise ValueError('Proposal violates native command interface')
        encoded=2*(target-self.limits[:,0])/np.diff(self.limits).ravel()-1
        output[self.case]=torch.as_tensor(encoded,device=student.device,dtype=student.dtype)
        realized=self.decode(output[self.case].detach().cpu().numpy())
        if np.max(abs(realized[12:]-self.previous[12:]))>.100001:raise ValueError('Float32 slew violation')
        self.records.append(dict(tick=tick,offset=offset,mode=mode,q=state.q.copy(),v=state.v.copy(),
            contacts=state.contacts.copy(),foot_force_world_n=force.copy(),student_action=base.copy(),
            final_action=output[self.case].detach().cpu().numpy().copy(),previous_target_rad=self.previous.copy(),
            requested_target_rad=target.copy(),final_target_rad=realized.copy()))
        self.applied+=1;self.previous=realized;return output

    def save(self):
        shapes=dict(q=(25,),v=(24,),contacts=(4,),foot_force_world_n=(4,3),student_action=(18,),
            final_action=(18,),previous_target_rad=(18,),requested_target_rad=(18,),final_target_rad=(18,))
        np.savez_compressed(self.output/'sequence_native_trace.npz',**{
            k:np.stack([r[k] for r in self.records]) if self.records else np.empty((0,)+shape)
            for k,shape in shapes.items()})
        result=dict(case=self.case,start_tick=self.start,planned_steps=self.steps,applied_steps=self.applied,
            query_match=self.query_check or dict(passed=False),early_termination_tick=self.early_termination_tick,
            full_feedback_sequence_executed=self.applied==self.steps,
            controller_kind='model_rejected_proposal_with_actual_state_feedback',model_accepted=False,
            native_execution=True,measured_state_feedback=True,online_mpc=False,
            deployable_risk_trigger=False,valid_expert_labels=False,automatic_promotion=False,
            bundle_sha256=self.manifest['bundle_sha256'],
            records=[{k:v for k,v in r.items() if k not in shapes} for r in self.records])
        path=self.output/'sequence_native_result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
        return result

    def close(self):
        pass
