"""One causal all18 command owner after observed racket/ball separation.

All first-episode cases use the same lifecycle and controller. No case outcome,
TRAIN number, absolute takeover time, student residual after takeover, or new
learning is used. Trial results still require native full-episode auditing.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.pure_control_recovery_v1 import PureControlRecoveryV1
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1,RecoveryCommandArbiterV1
from scripts.audit_joint_sequence_proposal_v1 import plain


class NativePureControlRecoveryV1:
    def __init__(self,env,limits,default,configuration,output,observer):
        self.env=env;self.output=Path(output);self.observer=observer
        cfg=json.loads(Path(configuration).read_text())
        self.limits=limits.detach().cpu().numpy().astype(float)
        if not np.allclose(self.limits,cfg['interface']['limits'],atol=1e-6,rtol=0.):raise ValueError('Changed physical joint limits')
        self.interface_config=cfg['interface'];self.nominal=default.detach().cpu().numpy().astype(float)
        self.controllers={};self.managers=[EventRecoveryManagerV1() for _ in range(env.num_envs)]
        self.arbiter=RecoveryCommandArbiterV1(self.limits);self.rows=[];self.last_tick=-1
        self.feet=[observer.names.index(n) for n in ['FL_foot','FR_foot','RL_foot','RR_foot']]
        self.nonfeet=[i for i in range(len(observer.names)) if i not in self.feet]
        self.applied=0;self.last_target=None

    def controller(self,lane):
        if lane not in self.controllers:
            interface=PositionPDInterface(**self.interface_config)
            self.controllers[lane]=PureControlRecoveryV1(interface,self.nominal)
        return self.controllers[lane]

    def action(self,tick,observation,student,finished):
        from quadsmash.tasks.stand.ball_env import PHYSICS
        if tick!=self.last_tick+1:raise ValueError('Control update must be continuous')
        self.last_tick=tick;e=self.env
        output=student.clone();forces=self.observer.read().detach().cpu().numpy()
        if not np.isfinite(forces).all():raise ValueError('Nonfinite ground-pair input')
        touching=e.shuttle_contact.data.force_matrix_w[:,0,0].norm(dim=-1)>PHYSICS['contact_force_threshold_n']
        tool=e.robot.data.body_link_pos_w[:,e.racket_body_id];quat=e.robot.data.body_link_quat_w[:,e.racket_body_id]
        separated=(~touching)&(e._tool_gap(e.shuttle.data.root_pos_w,tool,quat)>PHYSICS['separation_hysteresis_m'])
        eligible=(e.rally.had_contact&e.rally.released&separated&(~finished)).cpu().numpy()
        episodes=e.episode_index.cpu().numpy();clock=(e.episode_length_buf*e.step_dt).cpu().numpy()
        previous=e.targets[:,e.joint_ids].detach().cpu().numpy().astype(float)
        for lane in range(e.num_envs):
            if bool(finished[lane]):continue
            manager=self.managers[lane]
            # Before release we need no expensive full-body model. Initialize
            # the stateful lifecycle on the real first eligible observation.
            if not eligible[lane] and manager.memory.phase=='PREFIX':continue
            c=self.controller(lane);state=c.model.state_from_native(e,lane)
            state.previous_command_target=previous[lane].copy()
            r=state.root_rotation_world
            roll=np.arctan2(r[2,1],r[2,2]);pitch=np.arcsin(np.clip(-r[2,0],-1,1))
            phase=manager.update(episode=int(episodes[lane]),time_s=float(clock[lane]),
                had_racket_contact=bool(e.rally.had_contact[lane]),released=bool(e.rally.released[lane]),
                racket_touching=bool(touching[lane]),separated_now=bool(separated[lane]),
                ground_normal_force_n=forces[lane,self.feet,2],roll_pitch_rad=[roll,pitch],
                root_velocity_world=r@state.v[:3],root_angular_velocity_world=r@state.v[3:6],
                joint_velocity=state.joint_velocity,
                nonfoot_contact_now=bool(np.linalg.norm(forces[lane,self.nonfeet],axis=1).max()>1.))
            if not phase['owns_final_recovery_command']:continue
            state.contacts=phase['ground_contacts'].copy()
            state.measured_force_world=forces[lane,self.feet].copy()
            proposal,diagnostic=c.command(state,phase['phase'],float(clock[lane]))
            # Full18 target replaces the old complete action. No mixer runs
            # after this boundary; the original native PD receives this target.
            final,mapping=self.arbiter.select(None,proposal,previous[lane],owns_recovery=True)
            normalized=2*(final-self.limits[:,0])/np.diff(self.limits).ravel()-1.
            output[lane]=torch.as_tensor(normalized,device=output.device,dtype=output.dtype)
            roundtrip=self.limits[:,0]+.5*(output[lane].cpu().numpy()+1)*np.diff(self.limits).ravel()
            if np.max(abs(roundtrip-final))>2e-6:raise ValueError('Final command mapping drift')
            self.rows.append(dict(tick=tick,lane=lane,time_s=float(clock[lane]),episode=int(episodes[lane]),
                q=state.q.copy(),v=state.v.copy(),previous=previous[lane],requested=proposal,
                final=final,ground_force=forces[lane,self.feet],phase=phase,diagnostic=diagnostic))
            self.applied+=1
        if tick%40==0:
            temp=self.output/'pure_control_progress.tmp';temp.write_text(json.dumps(dict(control_tick=tick,
                owned_cases=len(self.controllers),applied_steps=self.applied,training=False))+'\n')
            temp.replace(self.output/'pure_control_progress.json')
        self.last_target=self.limits[:,0]+.5*(output.cpu().numpy()+1)*np.diff(self.limits).ravel()
        return output

    def verify_applied(self,finished_before):
        # At an episode reset targets are deliberately restored. Compare only
        # the still-current episode; the pre-step dispatch trace remains saved.
        e=self.env;actual=e.targets[:,e.joint_ids].cpu().numpy()
        mask=(~finished_before).cpu().numpy()&(e.episode_index.cpu().numpy()==0)
        if mask.any() and np.max(abs(actual[mask]-self.last_target[mask]))>2e-6:
            raise ValueError('Controller output was overwritten before physical execution')

    def save(self):
        keys=['tick','lane','time_s','episode','q','v','previous','requested','final','ground_force']
        path=self.output/'pure_control_commands.npz'
        with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**{k:np.asarray([r[k] for r in self.rows]) for k in keys})
        path.with_suffix('.tmp').replace(path)
        with (self.output/'pure_control_diagnostics.jsonl').open('w') as f:
            for r in self.rows:f.write(json.dumps(plain({k:r[k] for k in ['tick','lane','phase','diagnostic']}),allow_nan=False)+'\n')
        result=dict(applied_steps=self.applied,owned_cases=sorted(self.controllers),
            solver_accepted_steps=sum(r['diagnostic']['solver_accepted'] for r in self.rows),
            fallback_steps=sum(not r['diagnostic']['solver_accepted'] for r in self.rows),
            causal_release_entry=True,all18_owner=True,old_residual_after_takeover=False,
            recovery_until_episode_end=True,sequence_mpc=False,valid_expert_labels=False,
            trained=False,native_success_not_inferred=True)
        (self.output/'pure_control_result.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
