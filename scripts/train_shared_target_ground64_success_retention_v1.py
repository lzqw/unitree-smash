#!/usr/bin/env python3
"""Ground64 native PPO with persistent recovery and successful-action rehearsal.

Explicit recovery execution branch; native physics, task thresholds and limits remain.
"""
import argparse,copy,hashlib,json,signal,sys,time,traceback,importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--base-checkpoint',type=Path,required=True)
p.add_argument('--body-source-evaluation',type=Path,required=True)
p.add_argument('--ground-source-evaluation',type=Path,required=True)
p.add_argument('--original-teacher-evaluation',type=Path,required=True)
p.add_argument('--precision-receipt',type=Path,required=True)
p.add_argument('--config',type=Path,default=ROOT/'configs/g07_shared_target_ground64_success_retention_v1.json')
initialization_options=p.add_mutually_exclusive_group()
initialization_options.add_argument('--resume',type=Path)
initialization_options.add_argument('--initialize-actor',type=Path,
    help='Explicitly transfer a compatible canonical public actor into a fresh joint-reward learner; no critic/Adam/physics/RNG/counter resume')
initialization_options.add_argument('--initialize-centered-policy',type=Path,
    help='Transfer an independently measured outer actor on the same fixed base into a fresh learner; no critic/Adam/physics/RNG/counter resume')
initialization_options.add_argument('--initialize-centered-action-units',type=Path,
    help='Explicit fresh measured-actor transfer to leg0.06 units with witnessed physical mean checks; not strict resume')
p.add_argument('--centered-source-evaluation',type=Path,
    help='Required original TRAIN128 native evaluation of --initialize-centered-policy')
p.add_argument('--resize-ground',action='store_true',help='Explicitly fork a canonical learner onto a larger fresh ground batch; no old physical state restoration')
p.add_argument('--posthit-teacher-policy',type=Path,
    help='Optional portable residual actor used only after the public observed-hit flag; training remains PPO residual learning')
p.add_argument('--posthit-teacher-blend',type=float,default=0.0,
    help='Blend weight for the post-hit teacher residual in [0,1]; applied only to current public observation after observed hit')
p.add_argument('--teacher-all-phases',action='store_true',
    help='Use the declared portable teacher on every current public phase; intended only for collecting on-policy teacher-assisted rollouts')
p.add_argument('--posthit-teacher-recovery-phase',action='store_true',
    help='Also keep the portable post-hit teacher active in the public foot-contact recovery phase; opt-in diagnostic branch')
p.add_argument('--posthit-teacher-recovery-only',action='store_true',
    help='Use the portable post-hit teacher only in the public foot-contact recovery phase; do not alter airborne hit actions')
p.add_argument('--posthit-teacher-leg-only',action='store_true',
    help='In post-hit assistance, blend only the first 12 leg residual joints; preserve arm/racket residuals')
p.add_argument('--case-expert-bank',type=Path,
    help='Causal measured expert bank: select one expert from the first observed ball feature and hold it per episode')
p.add_argument('--case-expert-blend',type=float,default=0.0,
    help='Blend weight for the causal case expert bank in [0,1]; the exported learner does not embed this teacher')
p.add_argument('--case-expert-recovery-only',action='store_true',
    help='Apply the selected case expert only in the public foot-contact recovery phase')
p.add_argument('--disable-success-retention',action='store_true',
    help='Explicit ablation: skip the verified-success rehearsal optimizer for an execution-assistance isolation run')
p.add_argument('--num-envs',type=int,default=128)
p.add_argument('--updates',type=int,default=8)
p.add_argument('--seed',type=int,default=751001)
p.add_argument('--resume-learning-rate',type=float,default=None,
    help='Optional post-resume PPO learning rate override; keeps the strict resume contract unchanged')
from scripts.origin0_launcher_parser_v4 import add_origin0_launcher_args
add_origin0_launcher_args(p,AppLauncher);args=p.parse_args()
centered_initialization=(args.initialize_centered_policy is not None or args.initialize_centered_action_units is not None)
if centered_initialization!=(args.centered_source_evaluation is not None):
    p.error('Supply a centered initialization option together with --centered-source-evaluation')
if args.posthit_teacher_recovery_only and args.posthit_teacher_recovery_phase:
    p.error('Choose either --posthit-teacher-recovery-only or --posthit-teacher-recovery-phase')
if args.case_expert_blend < 0.0 or args.case_expert_blend > 1.0:
    p.error('--case-expert-blend must be in [0,1]')
if args.case_expert_blend > 0.0 and args.case_expert_bank is None:
    p.error('--case-expert-blend requires --case-expert-bank')
if args.case_expert_bank is not None and args.posthit_teacher_policy is not None:
    p.error('Use either --case-expert-bank or --posthit-teacher-policy for an isolated teacher trial')
app=AppLauncher(args).app

import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.contact_recovery_origin0_v5 import contact_recovery_origin0_cfg as full_racket_origin0_cfg
from quadsmash.rl.contact_recovery_source_v5 import snapshot_execution_sources
from quadsmash.tasks.stand.success_priority_training_env_v1 import SuccessPriorityTrainingEnvV1
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases
from quadsmash.control.full_racket_residual_reward_v4 import EVENT_WEIGHTS
from quadsmash.control.full_racket_origin0_residual_v4 import baseline_mode
from quadsmash.control.measured_joint_placement_reward_v4 import placement_settings,REWARD_VERSION
from quadsmash.control.success_priority_alignment_reward_v1 import alignment_settings,REWARD_VERSION as ALIGNMENT_REWARD_VERSION
from quadsmash.rl.native_ppo import NativeRunner
from quadsmash.rl.success_retention_v1 import attach_success_retention,SCHEMA as RETENTION_SCHEMA
from quadsmash.rl.failure_recovery_guidance_v1 import attach_failure_recovery_guidance,SCHEMA as FAILURE_GUIDANCE_SCHEMA
from quadsmash.rl.posthit_exploration_v1 import exploration_contract, register_posthit_policy
register_posthit_policy()
from quadsmash.rl.phase_gated_recovery_v1 import (
    register_phase_gated_recovery_policy, register_phase_gated_approach_recovery_policy)
register_phase_gated_recovery_policy()
register_phase_gated_approach_recovery_policy()
from quadsmash.rl.frozen_v2_approach_v1 import register_frozen_v2_approach_policy
register_frozen_v2_approach_policy()
from quadsmash.rl.prehit_teacher_prior_v1 import (
    PreHitTeacherPriorAdapterMixin, settings as prior_settings, SCHEMA as PRIOR_SCHEMA)
from scripts.diagnostic_curriculum_trace_v2 import (DiagnosticCurriculumEpisodeTraceRecorderV2 as EpisodeTraceRecorderV4,
    trace_settings,retention_settings,diagnostic_retention_settings,speed_diagnostic_retention_settings,SCHEMA as TRACE_SCHEMA)
from scripts.export_complete_demonstrations_v1 import classify as classify_curriculum
from quadsmash.rl.full_racket_origin0_residual_ppo_v4 import (ResidualTrainingInterrupted,save_completed_update_error_checkpoint)
from quadsmash.rl.full_racket_body_residual_ppo_v4 import ResidualRslAdapterV4,reward_contact_offset
from quadsmash.control.full_racket_body_residual_v4 import ResidualActionMixerV4
from quadsmash.control.full_racket_contact_normal_search_v4 import contact_normal_body
from quadsmash.control.shared_target_assistance_v6 import LandingAssistedController,RECOVERY_FIELDS
from quadsmash.control.persistent_recovery_assistance_v1 import PersistentRecoveryAssistanceV1,SCHEMA as RECOVERY_EXECUTION_SCHEMA
from quadsmash.control.reactive_landing_v1 import settings as landing_settings
from quadsmash.control.landing_recovery_reward_v1 import settings as landing_reward_settings
from quadsmash.control.support_policy_contract_v1 import policy_support_config,add_support_controller
from scripts.ground64_training_support_v1 import GROUND,verify_ground_source,FirstGroundForceMonitor
from scripts.ground_extent_diagnostic_v1 import observe_ground
from scripts.native_whole_body_admission_v1 import observe_native_whole_body
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from tensordict import TensorDict
import io
from quadsmash.evaluation_v4 import independent_episode
from quadsmash.usd_dependencies_v4 import audit_composed_asset


SHARED_SCHEMA='g07-public-shared-target-bias-v5'
RESIDUAL_SCHEMA='g07-shared-target-residual-ppo-v1'

def shared_source_controller(checkpoint,count,device):
    if checkpoint.get('schema') != SHARED_SCHEMA or not checkpoint.get('native_execution'):
        raise ValueError('Require the native shared-target source checkpoint')
    main=ROOT.parent/'unitree_smash'
    path=main/'reports/G07_full_racket_v5/current_intercept_diagnosis_20261027/native_factory.py'
    spec=importlib.util.spec_from_file_location('shared_target_training_factory',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.make_controller(checkpoint['source_public_policy'],count,device,
        bias_world_m=checkpoint['shared_target_bias']['bias_world_m'])

class SharedTargetResidualAdapter(PreHitTeacherPriorAdapterMixin, ResidualRslAdapterV4):
    def __init__(self,wrapped,base_checkpoint,joint_scale_rad,stop_requested=lambda:False,arm_target_rate_rad_s=5.,max_scale_rad=.25,reward_velocity_override=None,
                 recovery_local_time_override_s=None,recovery_actor_gain_override=None,recovery_thigh_bias_override=None,
                 recovery_front_back_thigh_bias_override=None,recovery_calf_bias_override=None,recovery_roll_feedback_gain_override=None,
                 recovery_blend_s=None,recovery_side_aware_roll_feedback_override=None,landing_assistance=None,support_assistance=None,
                 posthit_teacher_policy=None,posthit_teacher_blend=0.0,teacher_all_phases=False,
                 posthit_teacher_recovery_phase=False,posthit_teacher_recovery_only=False,
                 posthit_teacher_leg_only=False,case_expert_bank=None,case_expert_blend=0.0,
                 case_expert_recovery_only=False):
        if not isinstance(posthit_teacher_blend,(int,float)) or not 0.0 <= float(posthit_teacher_blend) <= 1.0:
            raise ValueError('posthit teacher blend must be in [0,1]')
        self.wrapped=wrapped; self.base_checkpoint=base_checkpoint
        self.posthit_teacher_blend=float(posthit_teacher_blend)
        self.teacher_all_phases=bool(teacher_all_phases)
        self.posthit_teacher_recovery_phase=bool(posthit_teacher_recovery_phase)
        self.posthit_teacher_recovery_only=bool(posthit_teacher_recovery_only)
        self.posthit_teacher_leg_only=bool(posthit_teacher_leg_only)
        self.posthit_teacher=None
        self.posthit_teacher_path=str(posthit_teacher_policy) if posthit_teacher_policy is not None else None
        if posthit_teacher_policy is not None:
            payload=torch.load(posthit_teacher_policy,map_location='cpu',weights_only=False)
            blob=payload.get('residual_actor_jit') if isinstance(payload,dict) else None
            if not isinstance(blob,(bytes,bytearray)):
                raise ValueError('posthit teacher policy must contain residual_actor_jit')
            self.posthit_teacher=torch.jit.load(io.BytesIO(blob),map_location=self.device).eval()
            for parameter in self.posthit_teacher.parameters(): parameter.requires_grad_(False)
        self.case_expert_bank_path=str(case_expert_bank) if case_expert_bank is not None else None
        self.case_expert_blend=float(case_expert_blend)
        self.case_expert_recovery_only=bool(case_expert_recovery_only)
        self.case_expert_features=None
        self.case_expert_indices=None
        self.case_expert_actors=[]
        self.case_expert_selected=None
        self.case_expert_selected_case=None
        if case_expert_bank is not None:
            manifest=json.loads(Path(case_expert_bank).read_text())
            if manifest.get('schema') != 'case-conditioned-native-expert-bank-v1':
                raise ValueError('Unexpected case expert bank schema')
            rows=manifest.get('cases',[])
            if len(rows) != 128:
                raise ValueError('Case expert bank must cover the fixed TRAIN128 panel')
            self.case_expert_features=torch.as_tensor([row['feature'] for row in rows],device=self.device,dtype=torch.float32)
            unique_paths=sorted(set(row.get('expert_policy') for row in rows if row.get('expert_policy')))
            path_to_index={}
            for path in unique_paths:
                payload=torch.load(path,map_location='cpu',weights_only=False)
                blob=payload.get('residual_actor_jit')
                if payload.get('schema') != RESIDUAL_SCHEMA or not payload.get('native_execution') or not isinstance(blob,(bytes,bytearray)):
                    raise ValueError('Case expert bank contains a non-native residual policy')
                path_to_index[path]=len(self.case_expert_actors)
                self.case_expert_actors.append(torch.jit.load(io.BytesIO(blob),map_location=self.device).eval())
            self.case_expert_indices=torch.as_tensor([path_to_index.get(row.get('expert_policy'),-1) for row in rows],
                device=self.device,dtype=torch.long)
            self.case_expert_selected=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
            self.case_expert_selected_case=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
        self.controller=shared_source_controller(base_checkpoint,self.num_envs,self.device)
        # Recovery assistance is an explicit training branch.  It modifies the
        # existing public standing handover parameters only; it does not read
        # rally labels, contact truth, or future simulator state.
        reference=self.controller.reference_controller
        recovery_overrides=dict(recovery_local_time_s=recovery_local_time_override_s,
            recovery_actor_gain=recovery_actor_gain_override,
            recovery_thigh_bias_rad=recovery_thigh_bias_override,
            recovery_front_back_thigh_bias_rad=recovery_front_back_thigh_bias_override,
            recovery_calf_bias_rad=recovery_calf_bias_override,
            recovery_roll_feedback_gain=recovery_roll_feedback_gain_override,
            recovery_blend_s=recovery_blend_s,
            recovery_side_aware_roll_feedback=recovery_side_aware_roll_feedback_override)
        self.controller=PersistentRecoveryAssistanceV1(self.controller,
            {k:v for k,v in recovery_overrides.items() if v is not None})
        if landing_assistance is not None:
            if support_assistance is not None:
                raise ValueError('Choose one declared landing assistance branch')
            self.controller=LandingAssistedController(self.controller,self.num_envs,self.device,
                landing_assistance,base_checkpoint['asset_sha256'])
        self.controller=add_support_controller(self.controller,self.num_envs,self.device,
            support_assistance,base_checkpoint['asset_sha256'])
        self.mixer=ResidualActionMixerV4(self.num_envs,self.device,joint_scale_rad,arm_target_rate_rad_s,max_scale_rad)
        env=self.unwrapped
        self.default=env.robot.data.default_joint_pos[0,env.joint_ids]
        self.limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
        self.stop_requested=stop_requested; self.total_control_env_steps=0
        self.last_record_count=len(env.rally_records); self.trace_recorder=None
        self.trace_iteration=0; self.trace_contact_cursor=len(env.contact_records)
        env.reward_offset.copy_(reward_contact_offset(reference.teacher.contact_offset,
            getattr(env,'live_alignment_reward',None) is not None))
        velocity = reference.teacher.velocity_world if reward_velocity_override is None else torch.as_tensor(
            reward_velocity_override,device=self.device,dtype=torch.float32)
        if velocity.shape != (3,) or not torch.isfinite(velocity).all():
            raise ValueError('Invalid training reward velocity override')
        env.reward_velocity.copy_(velocity)
        axes=torch.eye(3,device=self.device)
        n=contact_normal_body(reference.parameters,reference.teacher.normal_world,
            axes[0][None].expand(self.num_envs,-1),axes[1][None].expand(self.num_envs,-1),
            axes[2][None].expand(self.num_envs,-1))
        env.reward_normal.copy_(reference.teacher.face_sign*n[0])
        self.last_observation=self.wrapped.get_observations()
        self.controller.reset(); self.mixer.reset(); self._sync_reference()

    def _reset_case_expert(self,ids=None):
        if self.case_expert_selected is None:return
        if ids is None:
            self.case_expert_selected.fill_(-1);self.case_expert_selected_case.fill_(-1)
        else:
            self.case_expert_selected[ids]=-1;self.case_expert_selected_case[ids]=-1

    def case_expert_diagnostics(self):
        if self.case_expert_selected is None:return dict(enabled=False)
        selected=self.case_expert_selected_case.detach().cpu().tolist()
        return dict(enabled=True,selected_envs=sum(x>=0 for x in selected),
                    unique_selected_cases=sorted(set(x for x in selected if x>=0)),
                    expert_count=len(self.case_expert_actors),blend=self.case_expert_blend,
                    recovery_only=self.case_expert_recovery_only)

    def reset(self):
        result=super().reset()
        self._reset_case_expert()
        return result

    def step(self,residual):
        # Teacher assistance is causal: it uses only the current public
        # observation (optionally gated by its observed-hit flag).  The PPO
        # actor still supplies the learner action and remains the exported policy.
        effective=residual
        teacher_active=None
        if self.case_expert_selected is not None and self.case_expert_blend > 0.0:
            public=self.last_observation['policy']
            present=public[:,67] > 0.5
            choose=(self.case_expert_selected < 0) & present
            if bool(choose.any()):
                feature=torch.cat((public[choose,61:67],public[choose,759:770],public[choose,775:776]),dim=1)
                distance=((feature[:,None,:]-self.case_expert_features[None,:,:])**2).sum(-1)
                case=distance.argmin(1)
                chosen=self.case_expert_indices[case]
                self.case_expert_selected_case[choose]=case
                self.case_expert_selected[choose]=chosen
                missing=chosen < 0
                if bool(missing.any()):
                    chosen_ids=torch.where(choose)[0]
                    self.case_expert_selected[chosen_ids[missing]]=-2
            teacher_active=self.case_expert_selected >= 0
            if self.case_expert_recovery_only:
                teacher_active &= public[:,774] > 0.5
            if bool(teacher_active.any()):
                teacher_residual=torch.zeros_like(residual)
                for expert_index,actor in enumerate(self.case_expert_actors):
                    mask=self.case_expert_selected == expert_index
                    if bool(mask.any()):
                        with torch.no_grad():teacher_residual[mask]=actor(public[mask]).to(residual.device)
                weight=teacher_active.to(residual.dtype)[:,None]*self.case_expert_blend
                effective=(1.0-weight)*effective+weight*teacher_residual
        if self.posthit_teacher is not None and self.posthit_teacher_blend > 0.0:
            public=self.last_observation['policy']
            teacher_active=(torch.ones(len(public),dtype=torch.bool,device=public.device)
                            if self.teacher_all_phases else
                            ((public[:,774] > 0.5) if self.posthit_teacher_recovery_only else
                             ((public[:,773] > 0.5) |
                              (self.posthit_teacher_recovery_phase & (public[:,774] > 0.5)))))
            if bool(teacher_active.any()):
                with torch.no_grad():
                    teacher_residual=self.posthit_teacher(public).to(residual.device)
                if self.posthit_teacher_leg_only:
                    teacher_residual=teacher_residual.clone()
                    teacher_residual[:,12:]=residual[:,12:]
                weight=teacher_active.to(residual.dtype)[:,None] * self.posthit_teacher_blend
                effective=(1.0-weight)*residual + weight*teacher_residual
        result=super().step(effective)
        if self.case_expert_selected is not None:
            done_ids=torch.where(result[2].bool())[0]
            if len(done_ids):self._reset_case_expert(done_ids)
        if teacher_active is not None:
            self.last_teacher_active_count=int(teacher_active.sum().item())
        return result

def export_shared_policy(base_checkpoint,policy,joint_scale_rad,public_sample,metadata):
    if hasattr(policy, 'export_deterministic_actor'):
        actor=policy.export_deterministic_actor()
    else:
        actor=torch.nn.Sequential(copy.deepcopy(policy.actor_obs_normalizer).cpu().eval(),
                                  copy.deepcopy(policy.actor).cpu().eval())
    scripted=torch.jit.script(actor)
    # Exercise every raw phase even if the final rollout sample happens to
    # contain only standing lanes. A hard-coded export gate must be detected
    # before spending another native evaluation on a different policy.
    phase_probe=public_sample.repeat(5,1)
    phase_probe[:,770:775]=torch.eye(5,device=public_sample.device).repeat_interleave(len(public_sample),dim=0)
    public_sample=torch.cat((public_sample,phase_probe),dim=0)
    with torch.inference_mode():
        expected=policy.act_inference(TensorDict({'policy':public_sample},batch_size=[len(public_sample)])).cpu()
        actual=scripted.to(public_sample.device)(public_sample).cpu()
    err=float((expected-actual).abs().max())
    if not torch.allclose(actual,expected,atol=2e-5,rtol=2e-5): raise RuntimeError('Residual export mismatch')
    blob=io.BytesIO(); torch.jit.save(scripted.cpu(),blob); data=blob.getvalue()
    result=dict(schema=RESIDUAL_SCHEMA,parameters=base_checkpoint['parameters'].clone(),
        asset_sha256=base_checkpoint['asset_sha256'],base_controller=copy.deepcopy(base_checkpoint),
        residual_actor_jit=data,residual_joint_scale_rad=list(joint_scale_rad),
        residual_arm_target_rate_rad_s=float(metadata.get('arm_target_rate_rad_s',5.)),
        residual_scale_limit_rad=float(metadata.get('residual_scale_limit_rad',max(joint_scale_rad))),
        residual_base_mode='shared_target_bias',native_execution=bool(metadata.get('native_execution',False)),
        performance_inherited=False,formal_acceptance=False,
        shared_target_bias=copy.deepcopy(base_checkpoint['shared_target_bias']),
        protocol_sha256=base_checkpoint['protocol_sha256'],kind='Native residual PPO over frozen shared-target source',
        residual_training_metadata=dict(metadata,export_actor_max_error=err,actor_jit_sha256=hashlib.sha256(data).hexdigest(),
            actor_input='Current public794D observation only; no critic, case IDs or future labels.'))
    return result

stopping=False
def stop(*_):
    global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def atomic_torch(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def joined_episodes(env,cases,indices,protocol):
    assert len(env.rally_records)==len(env.episode_records),'Native rally/jump episode count mismatch'
    rows=[];checked=[];curriculum_checked=[]
    for rally,jump in zip(env.rally_records,env.episode_records):
        assert rally['env_id']==jump['env_id'],'Native rally/jump episode order mismatch'
        lane=rally['env_id'];index=indices[lane%len(indices)]
        row=dict(rally,jump=jump,case_id=f'TRAIN{index}_LANE{lane}_EP{rally["episode_index"]}',
                 source_training_index=index,ground_init=True)
        rows.append(row)
        checked.append(independent_episode(row,protocol))
        # The active G07 objective uses the authorized 5 m/s endpoint.  Keep
        # the original 8 m/s label for comparison, but report the curriculum
        # endpoint as the primary complete count so progress.json agrees with
        # the reward and trace-selection branches.
        curriculum_checked.append(classify_curriculum(row,protocol,5.0))
    return rows,dict(completed_episodes=len(rows),air=sum(v['airborne_contact'] for v in checked),
        conditional_recovery_count=sum(v['conditional_recovery'] for v in checked),
        original_complete8=sum(v['complete'] for v in checked),
        curriculum_complete5=sum(v['curriculum_complete'] for v in curriculum_checked),
        complete=sum(v['curriculum_complete'] for v in curriculum_checked),
        numerical_failures=sum(r['numerical_failures'] for r in rows))


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'INPUTS.json').exists():raise ValueError('Use a new native output directory')
    if args.updates<1:raise ValueError('At least one actual PPO update required')
    # The runner below has explicit, contract-checked paths for canonical
    # resume and actor-only transfer.  Keep those paths available: they reset
    # the physical environment at a ground boundary and never restore an
    # arbitrary mid-episode state.  Fresh training remains the default.
    if args.resize_ground and args.resume is None:raise ValueError('--resize-ground requires an actual canonical --resume checkpoint')
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol,None)
    if args.num_envs<len(cases) or args.num_envs%len(cases):
        raise ValueError('Use a positive multiple of all128 fixed TRAIN cases')
    configuration=json.loads(args.config.read_text());scales=configuration['joint_scale_rad']
    if configuration.get('recovery_assistance_execution') != RECOVERY_EXECUTION_SCHEMA:
        raise ValueError('Require explicit reset-persistent recovery training configuration')
    if configuration.get('ground_extent_diagnostic') != GROUND:
        raise ValueError('Require explicit ground64 provenance in the training recipe')
    ground_source=verify_ground_source(args.ground_source_evaluation,args.original_teacher_evaluation,args.base_checkpoint,protocol)
    atomic_json(args.output/'ground_source_verification.json',ground_source)
    scale_limit=float(configuration.get('residual_scale_limit_rad',.25))
    if (not np.isfinite(scale_limit) or not 0 < scale_limit <= .5
            or len(scales) != 18 or any((not isinstance(x,(int,float)) or not np.isfinite(x) or x <= 0 or x > scale_limit) for x in scales)
            or len(configuration.get('fresh_exploration_std',[])) != 18
            or any((not isinstance(x,(int,float)) or not np.isfinite(x) or x <= 0)
                   for x in configuration['fresh_exploration_std'])):
        raise ValueError('Require18 positive finite residual scales within the declared authority and exploration stds')
    arm_target_rate=float(configuration.get('arm_target_rate_rad_s',5.))
    if not np.isfinite(arm_target_rate) or not 0 < arm_target_rate <= 20.:
        raise ValueError('Invalid residual arm target rate')
    reward_velocity_override=configuration.get('reward_velocity_override_mps')
    if reward_velocity_override is not None:
        if (not isinstance(reward_velocity_override,list) or len(reward_velocity_override)!=3
                or any((not isinstance(x,(int,float)) or not np.isfinite(x)) for x in reward_velocity_override)):
            raise ValueError('Invalid reward velocity override')
    racket_speed_reward=configuration.get('racket_speed_reward')
    if racket_speed_reward is not None:
        if (not isinstance(racket_speed_reward,dict) or set(racket_speed_reward) != {'weight','target_mps'}
                or any((not isinstance(racket_speed_reward[k],(int,float)) or not np.isfinite(racket_speed_reward[k])
                        or racket_speed_reward[k] <= 0) for k in racket_speed_reward)
                or racket_speed_reward['weight'] > 20 or racket_speed_reward['target_mps'] > 20):
            raise ValueError('Invalid racket speed TRAIN shaping settings')
    attack_speed_threshold=float(configuration.get('attack_speed_threshold_mps',8.))
    landing_cfg=landing_settings(configuration.get('landing_assistance'))
    support_cfg=policy_support_config(configuration)
    if landing_cfg is not None and support_cfg is not None:
        raise ValueError('Choose one declared landing assistance branch')
    landing_reward_cfg=landing_reward_settings(configuration.get('landing_recovery_reward'))
    if not np.isfinite(attack_speed_threshold) or not 0 < attack_speed_threshold <= 8.:
        raise ValueError('Invalid TRAIN attack speed threshold')
    recovery_cfg=configuration.get('recovery_assistance',{})
    if not isinstance(recovery_cfg,dict):raise ValueError('Recovery assistance must be an object')
    recovery_allowed={'recovery_local_time_s','recovery_actor_gain','recovery_thigh_bias_rad',
        'recovery_front_back_thigh_bias_rad','recovery_calf_bias_rad','recovery_roll_feedback_gain',
        'recovery_blend_s','recovery_side_aware_roll_feedback'}
    if set(recovery_cfg)-recovery_allowed:raise ValueError('Unknown recovery assistance field')
    for key,value in recovery_cfg.items():
        if not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError('Recovery assistance values must be finite numbers')
    if 'recovery_local_time_s' in recovery_cfg and not 1.15 <= recovery_cfg['recovery_local_time_s'] <= 2.2:
        raise ValueError('Recovery handover must stay within the declared local-time range')
    if 'recovery_actor_gain' in recovery_cfg and not .20 <= recovery_cfg['recovery_actor_gain'] <= 1.20:
        raise ValueError('Recovery actor gain outside the declared standing range')
    if 'recovery_thigh_bias_rad' in recovery_cfg and not .05 <= recovery_cfg['recovery_thigh_bias_rad'] <= .45:
        raise ValueError('Recovery thigh bias outside the declared standing range')
    if 'recovery_front_back_thigh_bias_rad' in recovery_cfg and not -.15 <= recovery_cfg['recovery_front_back_thigh_bias_rad'] <= .15:
        raise ValueError('Recovery front/back thigh bias outside the declared standing range')
    if 'recovery_calf_bias_rad' in recovery_cfg and not -.45 <= recovery_cfg['recovery_calf_bias_rad'] <= .05:
        raise ValueError('Recovery calf bias outside the declared standing range')
    if 'recovery_roll_feedback_gain' in recovery_cfg and not -1. <= recovery_cfg['recovery_roll_feedback_gain'] <= 0.:
        raise ValueError('Recovery roll feedback outside the declared standing range')
    if 'recovery_blend_s' in recovery_cfg and not 0. <= recovery_cfg['recovery_blend_s'] <= .25:
        raise ValueError('Recovery blend outside the declared handover range')
    if 'recovery_side_aware_roll_feedback' in recovery_cfg and recovery_cfg['recovery_side_aware_roll_feedback'] not in (0, 1, 0., 1.):
        raise ValueError('Side-aware roll feedback must be 0 or 1')
    mode=baseline_mode(configuration.get('base_controller_mode'))
    base=torch.load(args.base_checkpoint,map_location='cpu',weights_only=False)
    source_verification=dict(shared_source=True,source_checkpoint_sha256=digest(args.base_checkpoint),
        source_evaluation=str(args.body_source_evaluation),formal_cases_read=False,formal_acceptance=False)
    if base.get('schema') != SHARED_SCHEMA or not base.get('native_execution'):
        raise ValueError('Require the native shared-target source')
    if base.get('protocol_sha256') != digest(protocol_path):
        raise ValueError('Shared source protocol differs from current protocol')
    source_job=json.loads((args.body_source_evaluation/'job.json').read_text())
    source_completed=json.loads((args.body_source_evaluation/'completed.json').read_text())
    source_metrics=json.loads((args.body_source_evaluation/'metrics.json').read_text())
    if source_job.get('status') != 'SUCCEEDED' or source_job.get('exit_code') != 0 or source_completed.get('env_steps') != 102400:
        raise ValueError('Require completed independent cold source evaluation')
    if source_metrics.get('history',[{}])[0].get('summary',{}).get('numerical_failures') != 0:
        raise ValueError('Shared source cold evaluation has numerical failures')
    if base['asset_sha256']!=protocol['asset']['sha256'] or digest(ROOT/protocol['asset']['path'])!=protocol['asset']['sha256']:
        raise ValueError('Baseline/full-model asset mismatch')
    for name,expected in protocol['physical_configs'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Physical config changed: '+name)
    joint_placement=placement_settings(configuration.get('joint_placement_reward'))
    live_alignment=alignment_settings(configuration.get('live_alignment_reward'))
    episode_trace=trace_settings(configuration.get('episode_trace'))
    retention=retention_settings(configuration.get('curriculum_trace'))
    failure_guidance=configuration.get('failure_recovery_guidance')
    if failure_guidance is not None:
        required_guidance={'manifest','manifest_sha256','learning_rate','steps_per_update','batch_size','anchor_weight','seed'}
        if not isinstance(failure_guidance,dict) or set(failure_guidance)!=required_guidance:
            raise ValueError('Declare the complete failure recovery guidance configuration')
    diagnostic_retention=diagnostic_retention_settings(configuration.get('diagnostic_trace'))
    speed_diagnostic_retention=speed_diagnostic_retention_settings(configuration.get('speed_diagnostic_trace'))
    if episode_trace is None:
        raise ValueError('Curriculum trace branch requires actual episode recording')
    if retention['minimum_speed_mps']!=attack_speed_threshold:
        raise ValueError('Retention speed must equal the explicit TRAIN curriculum threshold')
    if live_alignment is not None:
        if joint_placement is None:
            raise ValueError('Measured strings/attack shaping requires explicit joint placement')
        if live_alignment['discount']!=configuration['runner']['algorithm']['gamma']:
            raise ValueError('Potential discount must equal the actual learner gamma')
    if args.initialize_actor is not None and joint_placement is None:
        raise ValueError('--initialize-actor requires the explicit joint-placement reward config')
    contract=dict(base_policy_sha256=digest(args.base_checkpoint),protocol_sha256=digest(protocol_path),
        joint_scale_rad=scales,reward_version='native-full-chain-residual-v1',event_weights=EVENT_WEIGHTS,
        actor_warmup_updates=configuration['actor_warmup_updates'],seed=args.seed)
    contract['base_controller_mode']='shared_target_bias'
    contract['ground_extent_diagnostic']=copy.deepcopy(GROUND)
    contract['ground_source_verification_sha256']=digest(args.output/'ground_source_verification.json')
    if joint_placement is not None:
        contract.update(reward_version=REWARD_VERSION,joint_placement_reward=joint_placement)
    if live_alignment is not None:
        contract.update(reward_version=ALIGNMENT_REWARD_VERSION,live_alignment_reward=live_alignment,
            target_event_requires_actual_attack=True,fixed_upper_uv_position_shaping=False)
    contract.update(numerical_execution=protocol['numerical_execution'],
        origin_precision_receipt_sha256=digest(args.precision_receipt))
    contract.update(fresh_exploration_std=configuration['fresh_exploration_std'],base_controller_schema=base['schema'],fixed_source_actor_count=2,
        residual_scale_limit_rad=scale_limit,
        arm_target_rate_rad_s=arm_target_rate,
        reward_velocity_override_mps=reward_velocity_override,
        racket_speed_reward=racket_speed_reward,
        attack_speed_threshold_mps=attack_speed_threshold,
        recovery_assistance=recovery_cfg,
        recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
        allow_joint_scale_transfer=bool(configuration.get('allow_joint_scale_transfer',False)))
    priority_reward=configuration['success_priority_reward']
    if priority_reward != {'release_speed_saturation_mps': 5.0, 'independent_racket_speed_shaping_enabled': True}:
        raise ValueError('Require the user-authorized success-priority reward contract')
    if (attack_speed_threshold != 5.0 or not isinstance(racket_speed_reward, dict)
            or racket_speed_reward.get('target_mps') != 5.0):
        raise ValueError('Require 5 m/s task events and retained racket-speed reward capped at5m/s')
    contract['success_priority_reward']=copy.deepcopy(priority_reward)
    contract['primary_success_requirement_sha256']=digest(ROOT/'configs/g07_success_priority_speed5_v1.json')
    prior_cfg=prior_settings(configuration['prehit_teacher_prior'])
    contract['prehit_teacher_prior']=dict(schema=PRIOR_SCHEMA,**prior_cfg,public_phase_indices=[0,1,2],
        cost_scope='Requested clipped residual in radians, before range/slew limiting; current public phase only')
    contract['recovery_association']=protocol['recovery_association']
    posthit_teacher_assistance=dict(
        enabled=args.posthit_teacher_policy is not None and args.posthit_teacher_blend>0,
        policy=str(args.posthit_teacher_policy) if args.posthit_teacher_policy is not None else None,
        policy_sha256=digest(args.posthit_teacher_policy) if args.posthit_teacher_policy is not None else None,
        blend=float(args.posthit_teacher_blend),
        gate=('all_current_public_phases' if args.teacher_all_phases else
              ('public_observed_hit_or_recovery_index_773_or_774_gt_0.5'
               if args.posthit_teacher_recovery_phase else
               ('public_observed_recovery_index_774_gt_0.5'
                if args.posthit_teacher_recovery_only else 'public_observed_hit_index_773_gt_0.5'))),
        all_phases=bool(args.teacher_all_phases),
        recovery_phase_included=bool(args.posthit_teacher_recovery_phase),
        causal=True,exported_policy_does_not_embed_teacher=True)
    # Keep the legacy contract byte-compatible for ordinary post-hit runs so
    # checkpoints created before the recovery-only gate can still be resumed.
    if args.posthit_teacher_recovery_only:
        posthit_teacher_assistance['recovery_only']=True
    if args.posthit_teacher_leg_only:
        posthit_teacher_assistance['leg_only']=True
    contract['posthit_teacher_assistance']=posthit_teacher_assistance
    contract['case_expert_assistance']=dict(
        enabled=args.case_expert_bank is not None and args.case_expert_blend>0.0,
        bank=str(args.case_expert_bank) if args.case_expert_bank is not None else None,
        bank_sha256=digest(args.case_expert_bank) if args.case_expert_bank is not None else None,
        blend=float(args.case_expert_blend),
        recovery_only=bool(args.case_expert_recovery_only),
        causal_selection='first_observed_ball_feature_then_hold_until_reset',
        exported_policy_does_not_embed_teacher=True)
    contract['failure_recovery_guidance']=(dict(schema=FAILURE_GUIDANCE_SCHEMA,**failure_guidance)
        if failure_guidance is not None else None)
    if configuration.get('phase_gated_recovery') is not None:
        contract['phase_local_policy'] = copy.deepcopy(configuration['phase_gated_recovery'])
    if landing_cfg is not None:contract['landing_assistance']=landing_cfg
    if support_cfg is not None:contract['support_assistance']=support_cfg
    if landing_reward_cfg is not None:contract['landing_recovery_reward']=landing_reward_cfg
    contract['success_retention']=dict(schema=RETENTION_SCHEMA,**configuration['success_retention'])
    if args.disable_success_retention:
        contract['success_retention']['disabled']=True
    train_cfg=copy.deepcopy(configuration['runner']);train_cfg['seed']=args.seed;train_cfg['residual_contract']=contract
    if episode_trace is not None:train_cfg['episode_trace']=dict(schema=TRACE_SCHEMA,**episode_trace,curriculum_retention=retention,diagnostic_retention=diagnostic_retention,speed_diagnostic_retention=speed_diagnostic_retention)
    exploration = exploration_contract(train_cfg['policy'])
    if exploration is not None:contract['exploration']=exploration
    if train_cfg['obs_groups']['policy']!=['policy'] or train_cfg['policy']['noise_std_type']!='log':
        raise ValueError('This adapter requires the declared public-only feedforward log-std actor')
    execution_snapshot=snapshot_execution_sources(ROOT,args.output)
    cfg=full_racket_origin0_cfg(args.num_envs,protocol,args.precision_receipt);cfg.seed=args.seed
    cfg.episode_length_s=protocol['episode_seconds']
    original_ground_size=tuple(cfg.ground_size_m)
    if original_ground_size != tuple(GROUND['original_configured_m']):
        raise ValueError('Unexpected source ground dimensions')
    cfg.ground_size_m=tuple(GROUND['requested_m'])
    env=SuccessPriorityTrainingEnvV1(cfg,cases=cases*(args.num_envs//len(cases)),
        success_priority_reward=priority_reward,
        continue_after_miss=True,joint_placement_reward=joint_placement,live_alignment_reward=live_alignment,
        racket_speed_reward=racket_speed_reward,
        attack_speed_threshold_mps=attack_speed_threshold,
        landing_recovery_reward=landing_reward_cfg,
        contact_flight_recovery=protocol['recovery_association']['schema'])
    wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    adapter=SharedTargetResidualAdapter(wrapped,base,scales,lambda:stopping,arm_target_rate,scale_limit,reward_velocity_override,
        recovery_local_time_override_s=recovery_cfg.get('recovery_local_time_s'),
        recovery_actor_gain_override=recovery_cfg.get('recovery_actor_gain'),
        recovery_thigh_bias_override=recovery_cfg.get('recovery_thigh_bias_rad'),
        recovery_front_back_thigh_bias_override=recovery_cfg.get('recovery_front_back_thigh_bias_rad'),
        recovery_calf_bias_override=recovery_cfg.get('recovery_calf_bias_rad'),
        recovery_roll_feedback_gain_override=recovery_cfg.get('recovery_roll_feedback_gain'),
        recovery_blend_s=recovery_cfg.get('recovery_blend_s'),
        recovery_side_aware_roll_feedback_override=recovery_cfg.get('recovery_side_aware_roll_feedback'),
        landing_assistance=landing_cfg,
        support_assistance=support_cfg,posthit_teacher_policy=args.posthit_teacher_policy,
        posthit_teacher_blend=args.posthit_teacher_blend,teacher_all_phases=args.teacher_all_phases,
        posthit_teacher_recovery_phase=args.posthit_teacher_recovery_phase,
        posthit_teacher_recovery_only=args.posthit_teacher_recovery_only,
        posthit_teacher_leg_only=args.posthit_teacher_leg_only,
        case_expert_bank=args.case_expert_bank,case_expert_blend=args.case_expert_blend,
        case_expert_recovery_only=args.case_expert_recovery_only)
    adapter.configure_prehit_prior(prior_cfg)
    runner=None;completed_update_safe=False
    try:
        if env.robot.is_fixed_base:raise ValueError('Actual floating ground-start robot required')
        atomic_json(args.output/'asset_dependencies.json',audit_composed_asset(ROOT,cfg.robot.spawn.usd_path))
        if support_cfg is not None:
            from isaacsim.core.utils.stage import get_current_stage
            from quadsmash.control.support_admission_v1 import validate_native_support_interface
            atomic_json(args.output/'native_support_interface.json',
                validate_native_support_interface(env,get_current_stage()))
        atomic_json(args.output/'INPUTS.json',dict(contract=contract,training_indices=indices,
            execution_source=execution_snapshot,
            num_envs=args.num_envs,requested_updates=args.updates,runner=train_cfg,
            origin_precision_receipt_sha256=digest(args.precision_receipt),origin_spacing_m=0.,
            configuration_sha256=digest(args.config),resume_checkpoint_sha256=digest(args.resume) if args.resume else None,
            episode_trace_settings=episode_trace,curriculum_trace_settings=retention,diagnostic_trace_settings=diagnostic_retention,
            speed_diagnostic_trace_settings=speed_diagnostic_retention,
            frozen_baseline_source_verification=source_verification,
            initial_actor_checkpoint_sha256=digest(args.initialize_actor) if args.initialize_actor else None,
            initial_centered_policy_sha256=digest(args.initialize_centered_policy) if args.initialize_centered_policy else None,
            initial_centered_action_units_policy_sha256=digest(args.initialize_centered_action_units) if args.initialize_centered_action_units else None,
            centered_source_evaluation=str(args.centered_source_evaluation) if args.centered_source_evaluation else None,
            resize_ground_requested=args.resize_ground,
            physics_dt_s=cfg.sim.dt,policy_dt_s=env.step_dt,formal_cases_read=False,
            formal_acceptance=False,scope='Native TRAIN learning across changing weights; no frozen-policy performance.'))
        runner=NativeRunner(adapter,train_cfg,log_dir=str(args.output/'ppo'),device=env.device)
        if args.disable_success_retention:
            atomic_json(args.output/'success_retention_admission.json',dict(
                schema=RETENTION_SCHEMA,disabled=True,teacher_action_override=False,
                reason='explicit execution-assistance isolation run'))
        else:
            retention_receipt=attach_success_retention(runner,configuration['success_retention'],contract)
            atomic_json(args.output/'success_retention_admission.json',retention_receipt)
        if failure_guidance is not None:
            guidance_receipt=attach_failure_recovery_guidance(runner,failure_guidance,contract)
            atomic_json(args.output/'failure_recovery_guidance_admission.json',guidance_receipt)
        initialization=dict(mode='fresh_zero_mean_actor',num_envs=args.num_envs,initial_iteration=0)
        if mode=='frozen_public_policy':
            initialization.update(mode='fresh_zero_mean_actor_over_frozen_public_base',
                fixed_base_policy_sha256=digest(args.base_checkpoint),trainable_action_joints=18)
        if args.resume:
            previous=torch.load(args.resume,map_location='cpu',weights_only=False)
            if previous.get('infos',{}).get('residual_contract')!=contract:
                raise ValueError('Residual baseline/scales/rewards changed across resume')
            if args.resize_ground:
                from scripts.residual_ground_resize_v4 import load_resized_ground
                initialization=load_resized_ground(runner,args.resume)
                runner.success_retention.load_state_dict(previous['infos']['success_retention_state'])
            else:
                runner.load(str(args.resume),load_optimizer=True)
                if args.resume_learning_rate is not None:
                    if not np.isfinite(args.resume_learning_rate) or args.resume_learning_rate <= 0:
                        raise ValueError('--resume-learning-rate must be finite and positive')
                    runner.alg.learning_rate=float(args.resume_learning_rate)
                    for group in runner.alg.optimizer.param_groups:
                        group['lr']=float(args.resume_learning_rate)
                initialization=dict(mode='strict_canonical_ground_resume',num_envs=args.num_envs,
                                    initial_iteration=runner.current_learning_iteration)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        elif args.initialize_actor:
            from scripts.initialize_residual_actor_v4 import initialize_actor
            initialization=initialize_actor(runner,args.initialize_actor)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        elif args.initialize_centered_policy:
            from scripts.initialize_centered_policy_v4 import initialize_centered_policy
            initialization=initialize_centered_policy(runner,args.initialize_centered_policy,
                args.centered_source_evaluation,base,ROOT)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        elif args.initialize_centered_action_units:
            from scripts.initialize_centered_action_units_v4 import initialize_centered_action_units
            initialization=initialize_centered_action_units(runner,args.initialize_centered_action_units,
                args.centered_source_evaluation,base,ROOT)
            adapter.controller.reset();adapter.mixer.reset();adapter._sync_reference()
        else:
            from quadsmash.rl.posthit_exploration_v1 import initialize_shared_target_actor
            initialization.update(initialize_shared_target_actor(runner.alg.policy,configuration))
            if runner.alg.optimizer.state:raise ValueError('Fresh zero actor must keep empty Adam state')
        if getattr(runner.alg.policy, 'freeze_verified_policy', False):
            policy=runner.alg.policy
            frozen_receipt=policy.capture_frozen_reference()
            if args.initialize_actor is not None:
                source=torch.load(args.initialize_actor,map_location='cpu',weights_only=False)
                old=torch.jit.load(io.BytesIO(source['residual_actor_jit']),map_location=env.device).eval()
                sample=runner.success_retention.public
                error=0.0
                with torch.no_grad():
                    for start in range(0,len(sample),2048):
                        public=sample[start:start+2048]
                        obs=TensorDict({'policy':public},batch_size=[len(public)])
                        error=max(error,float((policy.act_inference(obs)-old(public)).abs().max()))
                if error > 2e-5:
                    raise ValueError('New head changed the complete initial v2 policy: '+str(error))
                frozen_receipt.update(initial_source_policy_sha256=digest(args.initialize_actor),
                    initial_mapping_max_error=error,initial_mapping_probe_rows=len(sample),
                    full_v2_mapping_checked=True)
            atomic_json(args.output/'frozen_policy_admission.json',frozen_receipt)
        if not args.resume:
            atomic_torch(args.output/'fresh_initial_model_state.pt',dict(
                model_state_dict={name:value.detach().cpu().clone() for name,value in runner.alg.policy.state_dict().items()},
                base_policy_sha256=digest(args.base_checkpoint),actual_control_env_steps=adapter.total_control_env_steps,
                optimizer_entries=len(runner.alg.optimizer.state),fixed_source_residual_actor_count=2))
        input_record=json.loads((args.output/'INPUTS.json').read_text())
        input_record['initialization']=initialization
        atomic_json(args.output/'INPUTS.json',input_record)
        from isaacsim.core.utils.stage import get_current_stage
        atomic_json(args.output/'ground_geometry.json',observe_ground(get_current_stage(),original_ground_size,cfg.ground_size_m))
        admission,_interface=observe_native_whole_body(env,get_current_stage(),LoadedPinocchioModelV1())
        atomic_json(args.output/'native_admission.json',admission)
        if not admission['passed']:raise ValueError('Actual loaded model/actuator admission failed')
        if env.scene.env_origins.any():raise ValueError('Require original independent origin0 layout')
        if getattr(env,'_diagnostic_observer',None) is not None:raise ValueError('Preserve existing physical observer')
        ground_monitor=FirstGroundForceMonitor(env)
        env._diagnostic_observer=ground_monitor
        effective_recovery={}
        reference=adapter.controller.reference_controller
        for name,value in recovery_cfg.items():
            index=RECOVERY_FIELDS[name][0]
            if name == 'recovery_side_aware_roll_feedback':
                actual=float(reference.teacher.standing.side_aware_roll_feedback)
                matches=bool(actual == float(value))
            elif index is None:
                actual=reference.teacher.leg_recovery_blend.seconds
                matches=abs(actual-float(value))<1e-6
            else:
                actual=float(reference.parameters[0,index])
                matches=bool(torch.allclose(reference.parameters[:,index],
                    torch.full_like(reference.parameters[:,index],float(value)),atol=1e-6,rtol=0.))
            if not matches:raise ValueError('Recovery setting lost during native initialization: '+name)
            effective_recovery[name]=actual
        atomic_json(args.output/'recovery_parameters_after_reset.json',dict(
            schema=RECOVERY_EXECUTION_SCHEMA,requested=recovery_cfg,effective=effective_recovery,
            verified_lanes=args.num_envs,all_lanes_match=True,formal_acceptance=False))
        data=env.robot.data;ground_root=data.root_state_w.detach().cpu().clone()
        ground_root[:,:3]-=env.scene.env_origins.cpu()
        expected_root=torch.tensor(protocol['initialization']['root_position_m'],dtype=ground_root.dtype)
        if not torch.allclose(ground_root[:,:3],expected_root[None].expand(args.num_envs,-1),atol=1e-5,rtol=0):
            raise ValueError('Fresh native ground positions differ from the original protocol')
        if env.episode_length_buf.any():raise ValueError('Training must begin at actual episode clock zero')
        atomic_json(args.output/'initial_ground_states.json',dict(scope='Actual native state before this job steps; no physical continuation claim across batch resizing.',
            initialization=initialization,source_training_indices=[indices[i%len(indices)] for i in range(args.num_envs)],
            root_state_local=ground_root.tolist(),joint_position_rad=data.joint_pos[:,env.joint_ids].cpu().tolist(),
            joint_velocity_rad_s=data.joint_vel[:,env.joint_ids].cpu().tolist(),all_episode_clocks_zero=True))
        if episode_trace is not None:
            from isaacsim.core.utils.stage import get_current_stage
            from quadsmash.tasks.stand.visual_geometry import export_robot_geometry
            export_robot_geometry(get_current_stage(),env.robot,args.output)
            adapter.trace_recorder=EpisodeTraceRecorderV4(args.output/'trajectory_traces',args.num_envs,
                round(protocol['episode_seconds']/protocol['policy_dt_s'])+1,env.robot.body_names,
                episode_trace,dict(native_source_confirmed=True,source_kind='Actual native changing-policy TRAIN',
                    training_indices=indices,initialization=initialization,ground_extent_diagnostic=GROUND,
                    protocol_sha256=contract['protocol_sha256'],asset_sha256=protocol['asset']['sha256'],
                    body_names=env.robot.body_names,joint_names=[env.robot.joint_names[j]for j in env.joint_ids],
                    default_joint_position_rad=adapter.default.cpu().tolist(),joint_limits_rad=adapter.limits.cpu().tolist(),
                    source_job_output=str(args.output),formal_acceptance=False),retention=retention,diagnostic_retention=diagnostic_retention,speed_diagnostic_retention=speed_diagnostic_retention)
            adapter.trace_contact_cursor=len(env.contact_records)
        initial_update=runner.current_learning_iteration;history=[];interrupted=False
        for _ in range(args.updates):
            adapter.trace_iteration=runner.current_learning_iteration
            train_actor=runner.current_learning_iteration>=configuration['actor_warmup_updates']
            policy=runner.alg.policy
            if getattr(policy, 'phase_gated_recovery', False):
                # The policy selects the explicitly trainable phase-local
                # module. FrozenV2Approach keeps both old actor and recovery
                # head frozen; the legacy recovery branch opens its own head.
                policy.set_recovery_trainable(train_actor)
                policy.log_std.requires_grad_(False)
            else:
                for parameter in policy.actor.parameters():parameter.requires_grad_(train_actor)
                policy.log_std.requires_grad_(train_actor)
            completed_update_safe=False
            try:runner.learn(1,init_at_random_ep_len=False)
            except ResidualTrainingInterrupted:
                interrupted=True;break
            completed_update_safe=True
            rows,counts=joined_episodes(env,cases,indices,protocol)
            entry=dict(update=runner.current_learning_iteration,actor_warmup=not train_actor,
                actual_control_env_steps=adapter.total_control_env_steps,**counts,
                losses=runner.alg.last_update_stats,formal_acceptance=False,
                prehit_teacher_prior=adapter.prehit_prior_diagnostics(clear=True),
                case_expert_assistance=adapter.case_expert_diagnostics(),
                ground_contact_screen=ground_monitor.diagnostics())
            if getattr(policy, 'freeze_verified_policy', False):
                entry['frozen_policy_invariant']=policy.assert_frozen_reference()
            if env.live_alignment_reward is not None:
                entry['actual_paid_alignment_rewards']=env.live_alignment_reward.statistics()
            if env.landing_recovery_reward is not None:
                entry['actual_paid_landing_rewards']=env.landing_recovery_reward.statistics()
            if adapter.trace_recorder is not None:
                try:adapter.trace_recorder.flush()
                except Exception as error:adapter.trace_recorder.disable(error)
                entry['trajectory_recording']=adapter.trace_recorder.statistics()
            history.append(entry);atomic_json(args.output/'progress.json',entry)
            atomic_json(args.output/'episodes.partial.json',rows)
            print(json.dumps(entry),flush=True)
            # Plans are diagnostics, not policy state. Flush before clearing
            # long TRAIN logs; per-lane planning state remains intact.
            atomic_json(args.output/f'plans_update_{runner.current_learning_iteration:04d}.json',adapter.controller.logs)
            adapter.controller.logs=[]
            if entry['ground_contact_screen']['first_contact_downward_events']:
                atomic_json(args.output/'contact_anomaly.json',dict(update=runner.current_learning_iteration,
                    screen=entry['ground_contact_screen'],reason='Diagnose first-contact downward forces before spending further training updates'))
                interrupted=True;break
            if stopping:interrupted=True;break
        rows,counts=joined_episodes(env,cases,indices,protocol)
        public_sample=adapter.last_observation['policy'][:16].detach().clone()
        # Canonical reset is checkpoint bookkeeping, outside TRAIN monitoring.
        env._diagnostic_observer=None
        runner.save_restart(args.output/'checkpoint.pt',infos=dict(residual_contract=contract,
            learner_initialization=initialization))
        assert env.contact_recovery is not None
        if any(bool(value.any()) for value in vars(env.contact_recovery).values() if torch.is_tensor(value)):
            raise ValueError('Canonical ground restart retains contact-flight episode state')
        if env.landing_recovery_reward is not None and bool(env.landing_recovery_reward.credited.any()):
            raise ValueError('Canonical ground restart retains paid landing progress')
        atomic_json(args.output/'canonical_contact_flight_state.json',dict(
            all_contact_flight_buffers_zero=True,all_episode_clocks_zero=not bool(env.episode_length_buf.any()),
            checkpoint_sha256=digest(args.output/'checkpoint.pt'),arbitrary_mid_episode_restore_claimed=False))
        if adapter.trace_recorder is not None:
            try:adapter.trace_recorder.flush()
            except Exception as error:adapter.trace_recorder.disable(error)
        policy=export_shared_policy(base,runner.alg.policy,scales,public_sample,
            dict(native_execution=adapter.total_control_env_steps>0,completed_updates=runner.current_learning_iteration,
                 arm_target_rate_rad_s=arm_target_rate,
                 residual_scale_limit_rad=scale_limit,
                 reward_velocity_override_mps=reward_velocity_override,
                 racket_speed_reward=racket_speed_reward,
                 attack_speed_threshold_mps=attack_speed_threshold,
                 recovery_assistance=recovery_cfg,
                 recovery_assistance_execution=RECOVERY_EXECUTION_SCHEMA,
                 **({'landing_assistance':landing_cfg} if landing_cfg is not None else {}),
                 **({'support_assistance':support_cfg} if support_cfg is not None else {}),
                 **({'landing_recovery_reward':landing_reward_cfg} if landing_reward_cfg is not None else {}),
                 base_policy_sha256=contract['base_policy_sha256'],actual_control_env_steps=adapter.total_control_env_steps,
                 origin_precision_receipt_sha256=contract['origin_precision_receipt_sha256'],
                 **({'base_controller_mode':mode} if mode=='frozen_public_policy' else {}),
                 num_envs=args.num_envs,initialization=initialization,
                 prehit_teacher_prior=contract['prehit_teacher_prior'],
                 success_retention=contract['success_retention'],
                 failure_recovery_guidance=contract['failure_recovery_guidance'],
                 **({'phase_local_policy':contract['phase_local_policy']}
                    if 'phase_local_policy' in contract else {}),
                 posthit_teacher_assistance=contract['posthit_teacher_assistance'],
                 case_expert_assistance=contract['case_expert_assistance'],
                 ground_extent_diagnostic=copy.deepcopy(GROUND),
                 ground_source_verification_sha256=contract['ground_source_verification_sha256'],
                 exploration=contract['exploration'],actor_warmup_updates=configuration['actor_warmup_updates']))
        policy.update(protocol_sha256=contract['protocol_sha256'],
            recovery_association=protocol['recovery_association'],
            source_checkpoint_protocol_sha256=base['protocol_sha256'])
        atomic_torch(args.output/'policy.pt',policy)
        atomic_json(args.output/'episodes.json',rows)
        atomic_json(args.output/'contacts.json',env.contact_records)
        atomic_json(args.output/'metrics.json',dict(scope='Changing-policy native TRAIN rollouts only; evaluate exported policy separately.',
            completed_updates=runner.current_learning_iteration-initial_update,history=history,counts=counts,initialization=initialization,
            checkpoint_aborted_episodes=env.checkpoint_aborted_episodes,
            ground_extent_diagnostic=GROUND,ground_contact_screen=ground_monitor.diagnostics(),
            policy_sha256=digest(args.output/'policy.pt'),formal_acceptance=False))
        if interrupted:
            atomic_json(args.output/'interrupted.json',dict(env_steps=adapter.total_control_env_steps,
                resume_from='checkpoint.pt',canonical_ground_restart=True,formal_acceptance=False))
            raise SystemExit(2)
        assert runner.current_learning_iteration-initial_update==args.updates
        atomic_json(args.output/'completed.json',dict(env_steps=adapter.total_control_env_steps))
    except BaseException as error:
        env._diagnostic_observer=None
        if runner is not None and completed_update_safe:
            try:
                save_completed_update_error_checkpoint(runner,args.output,contract,initialization,
                    error,adapter.total_control_env_steps,completed_update_safe)
            except BaseException as save_error:
                atomic_json(args.output/'error_recovery_checkpoint_failed.json',dict(
                    original_error=repr(error),save_error=repr(save_error),job_succeeded=False))
        raise
    finally:
        env._diagnostic_observer=None
        adapter.close()


if __name__=='__main__':
    try:main()
    except BaseException:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise
    finally:app.close(skip_cleanup=True)
