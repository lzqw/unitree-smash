"""Query-matched native diagnostic of an offline recovery sequence.

This uses a recorded case/time, not a deployable risk trigger. It never resets
or teleports a state. The original student runs before and after the sequence;
post-sequence arm handoff obeys the original command-history slew bound.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pinocchio as pin
import torch
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


class NativeRecoverySequenceReplay:
    def __init__(self,bundle,case,limits,policy_sha256,output):
        self.bundle=Path(bundle);self.output=Path(output);self.case=int(case)
        self.manifest=json.loads((self.bundle/'manifest.json').read_text())
        m=self.manifest;archive=self.bundle/m['bundle_name']
        if (hashlib.sha256(archive.read_bytes()).hexdigest()!=m['bundle_sha256'] or
                m['policy_sha256']!=policy_sha256 or m['native_execution'] or m['valid_expert_labels']):
            raise ValueError('Require exact unexecuted, independently checked model bundle')
        if 'ground_geometry_sha256' not in m:raise ValueError('Require geometry-constrained export')
        for path,digest in m['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Bundle source changed '+path)
        matches=[r for r in m['rows'] if r['case']==self.case]
        if len(matches)!=1:raise ValueError('Selected candidate missing or duplicated')
        self.row=matches[0];self.start=int(self.row['query_tick'])
        self.limits=np.asarray(limits.detach().cpu(),dtype=float)
        if not np.allclose(self.limits,m['configuration']['interface']['limits'],atol=1e-7,rtol=0):
            raise ValueError('Native action limits differ')
        self.dt=float(m['control_dt_s']);self.arm_rate=float(m['configuration']['arm_target_rate_rad_s'])
        if self.dt!=.01 or self.arm_rate!=10.:raise ValueError('Unexpected original command contract')
        key=f'case_{self.case:03d}'
        with np.load(archive,allow_pickle=False) as data:
            self.commands=data[key+'_commands_rad'].copy()
            self.expected_q=data[key+'_predicted_q'].copy();self.expected_v=data[key+'_predicted_v'].copy()
            self.expected_contacts=data[key+'_predicted_contacts'].copy()
            self.query_public=data[key+'_query_public'].copy()
            self.query_previous_action=data[key+'_previous_action'].copy()
        if self.commands.shape!=(40,18) or not np.isfinite(self.commands).all():raise ValueError('Require full 40-step plan')
        if self.expected_q.shape!=(41,25) or self.expected_v.shape!=(41,24):raise ValueError('Invalid prediction shape')
        previous=self.decode(self.query_previous_action)
        sequence=np.vstack([previous,self.commands])
        if (np.any(self.commands<self.limits[:,0]-1e-7) or np.any(self.commands>self.limits[:,1]+1e-7) or
                np.max(abs(np.diff(sequence[:,12:],axis=0)))>self.arm_rate*self.dt+1e-6):
            raise ValueError('Stored commands violate original range/history interface')
        self.model=LoadedPinocchioModelV1();self.records=[];self.previous=None;self.last_tick=-1
        self.query_check=None;self.started=False;self.applied=0
        self.tolerances=dict(configuration_tangent=2e-5,velocity=2e-4,
            public=5e-4,previous_target_rad=5e-6)

    def decode(self,action):return self.limits[:,0]+.5*(np.asarray(action)+1.)*np.diff(self.limits).ravel()

    def action(self,tick,public,student,finished,native_state=None):
        if public.shape!=(128,794) or student.shape!=(128,18) or finished.shape!=(128,):
            raise ValueError('Require the unchanged full128 development panel')
        if tick!=self.last_tick+1:raise ValueError('Continuous real control ticks required')
        self.last_tick=tick;output=student.clone();raw=public[self.case].detach().cpu().numpy()
        base=student[self.case].detach().cpu().numpy();prior=None if self.previous is None else self.previous.copy()
        if tick<self.start:
            self.previous=self.decode(base);return output
        if bool(finished[self.case]):
            if not self.started or self.applied<40:raise ValueError('Selected native episode ended before full sequence')
            return output
        if native_state is None:raise ValueError('Live articulation state required after query')
        measured_force=np.asarray(getattr(native_state,'measured_force_world',None))
        if measured_force.shape!=(4,3) or not np.isfinite(measured_force).all():
            raise ValueError('Measured native foot forces required')
        if abs(float(raw[775])-tick*self.dt)>1e-5:raise ValueError('Original first-episode clock changed')
        offset=tick-self.start
        if offset==0:
            configuration=pin.difference(self.model.model,self.expected_q[0],native_state.q)
            errors=dict(configuration_tangent=float(np.max(abs(configuration))),
                velocity=float(np.max(abs(native_state.v-self.expected_v[0]))),
                public=float(np.max(abs(raw-self.query_public))),
                previous_target_rad=float(np.max(abs(prior-self.decode(self.query_previous_action)))))
            contact_equal=bool(np.array_equal(native_state.contacts,self.expected_contacts[0]))
            passed=contact_equal and all(errors[k]<=v for k,v in self.tolerances.items())
            self.query_check=dict(passed=passed,errors=errors,tolerances=self.tolerances,contacts_equal=contact_equal)
            (self.output/'sequence_query_match.json').write_text(json.dumps(self.query_check,indent=2)+'\n')
            if not passed:raise ValueError('Actual query/previous command differs; candidate not executed')
            self.started=True
        if offset<40:
            target=self.commands[offset].copy();mode='planned_sequence';self.applied+=1
        else:
            # This is an explicitly recorded handoff back to the student,
            # not a claimed MPC feedback tail. Only its arm rate is bounded
            # against what was actually dispatched, not hidden student history.
            target=self.decode(base);target[12:]=np.clip(target[12:],
                prior[12:]-self.arm_rate*self.dt,prior[12:]+self.arm_rate*self.dt)
            mode='student_with_command_history_handoff'
        encoded=2.*(target-self.limits[:,0])/np.diff(self.limits).ravel()-1.
        if not np.isfinite(encoded).all() or np.max(abs(encoded))>1.+1e-7:raise ValueError('Invalid final command')
        output[self.case]=torch.as_tensor(encoded,device=student.device,dtype=student.dtype)
        realized=self.decode(output[self.case].detach().cpu().numpy())
        if np.max(abs(realized[12:]-prior[12:]))>self.arm_rate*self.dt+1e-6:
            raise ValueError('Actual float32 dispatch violates arm command continuity')
        prediction_error=None
        if offset<=40:
            delta=pin.difference(self.model.model,self.expected_q[offset],native_state.q)
            prediction_error=dict(configuration_tangent_max=float(np.max(abs(delta))),
                joint_position_max_rad=float(np.max(abs(native_state.joint_position-self.expected_q[offset,self.model.q_indices]))),
                root_position_m=float(np.linalg.norm(native_state.q[:3]-self.expected_q[offset,:3])),
                root_attitude_rad=float(np.linalg.norm(delta[3:6])),
                velocity_max=float(np.max(abs(native_state.v-self.expected_v[offset]))),
                contacts_equal=bool(np.array_equal(native_state.contacts,self.expected_contacts[offset])))
        self.records.append(dict(control_tick=tick,offset=offset,mode=mode,
            q=native_state.q.copy(),v=native_state.v.copy(),contacts=native_state.contacts.copy(),
            foot_force_world_n=measured_force.copy(),
            student_action=base.copy(),final_action=output[self.case].detach().cpu().numpy().copy(),
            previous_target_rad=prior,requested_target_rad=target,final_target_rad=realized,
            prediction_error=prediction_error))
        self.previous=realized;return output

    def save(self):
        if not self.started or self.applied!=40:raise ValueError('Full planned sequence was not executed')
        numeric=['q','v','contacts','foot_force_world_n','student_action','final_action',
            'previous_target_rad','requested_target_rad','final_target_rad']
        np.savez_compressed(self.output/'sequence_native_trace.npz',
            **{k:np.stack([r[k] for r in self.records]) for k in numeric})
        result=dict(case=self.case,start_tick=self.start,planned_steps=40,applied_steps=self.applied,
            query_match=self.query_check,records=[{k:v for k,v in r.items() if k not in numeric} for r in self.records],
            bundle_sha256=self.manifest['bundle_sha256'],native_execution=True,
            case_time_conditioned_diagnostic=True,deployable_risk_trigger=False,
            post_sequence='original student with actual-history arm slew',
            sustained_mpc_feedback_implemented=False,valid_expert_labels=False)
        (self.output/'sequence_native_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
