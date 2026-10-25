"""Stock-RSL adapter and portable actor export for full-racket residual PPO."""
import copy
import hashlib
import io
import torch
from tensordict import TensorDict

from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.full_racket_residual_v4 import (
    ResidualActionMixerV4,FullRacketResidualControllerV4,SCHEMA,baseline_mode,validate_frozen_baseline)
from quadsmash.control.full_racket_contact_normal_search_v4 import contact_normal_body


class ResidualTrainingInterrupted(Exception):
    pass


def reward_contact_offset(teacher_offset,live_alignment_enabled):
    """Keep the new reward at the real strings center without editing teacher commands."""
    if not live_alignment_enabled:return teacher_offset
    result=teacher_offset.clone();result[1:]=0.
    return result


class ResidualRslAdapterV4:
    def __init__(self,wrapped,base_checkpoint,joint_scale_rad,stop_requested=lambda:False,base_mode=None):
        self.wrapped=wrapped;self.base_checkpoint=base_checkpoint
        mode=baseline_mode(base_mode)
        if mode=='frozen_public_policy':
            validate_frozen_baseline(base_checkpoint)
            self.controller=FullRacketResidualControllerV4(base_checkpoint,self.num_envs,self.device)
            for parameter in self.controller.actor.parameters():parameter.requires_grad_(False)
        else:
            self.controller=FullRacketRallyControllerV4(base_checkpoint,self.num_envs,self.device)
        self.mixer=ResidualActionMixerV4(self.num_envs,self.device,joint_scale_rad)
        env=self.unwrapped
        self.default=env.robot.data.default_joint_pos[0,env.joint_ids]
        self.limits=env.robot.data.joint_pos_limits[0,env.joint_ids]
        self.stop_requested=stop_requested;self.total_control_env_steps=0
        self.last_record_count=len(env.rally_records)
        self.trace_recorder=None;self.trace_iteration=0;self.trace_contact_cursor=len(env.contact_records)
        reference=getattr(self.controller,'reference_controller',self.controller)
        env.reward_offset.copy_(reward_contact_offset(reference.teacher.contact_offset,
            getattr(env,'live_alignment_reward',None) is not None))
        env.reward_velocity.copy_(reference.teacher.velocity_world)
        axes=torch.eye(3,device=self.device)
        n=contact_normal_body(reference.parameters,reference.teacher.normal_world,
            axes[0][None].expand(self.num_envs,-1),axes[1][None].expand(self.num_envs,-1),
            axes[2][None].expand(self.num_envs,-1))
        env.reward_normal.copy_(reference.teacher.face_sign*n[0])
        self.last_observation=self.wrapped.get_observations()
        self.controller.reset();self.mixer.reset();self._sync_reference()

    def __getattr__(self,name):return getattr(self.wrapped,name)

    @property
    def unwrapped(self):return self.wrapped.unwrapped

    def _sync_reference(self):
        env=self.unwrapped
        reference=getattr(self.controller,'reference_controller',self.controller)
        target=torch.as_tensor(reference.nominal_target,device=self.device,dtype=torch.float32)
        env.reward_target.copy_(target+reference.target_offset_world)
        env.reward_event_time.copy_(reference.event_time)
        env.reward_plan_valid.copy_(reference.planned)

    def reset(self):
        if self.trace_recorder is not None:
            try:self.trace_recorder.reset()
            except Exception as error:self.trace_recorder.disable(error)
        observation,extras=self.wrapped.reset()
        self.controller.reset();self.mixer.reset();self._sync_reference()
        self.last_observation=observation
        return observation,extras

    def get_observations(self):
        self.last_observation=self.wrapped.get_observations()
        return self.last_observation

    def step(self,residual):
        if self.stop_requested():raise ResidualTrainingInterrupted('Save at the last completed PPO update')
        base=self.controller.action(self.last_observation,self.default,self.limits)
        self._sync_reference()
        self.unwrapped.residual_raw.copy_(residual.detach().clamp(-1,1))
        actual=self.mixer.action(base,residual,self.last_observation['policy'],self.default,self.limits)
        if self.trace_recorder is not None and self.trace_recorder.disabled_error is None:
            try:
                env=self.unwrapped;data=env.robot.data
                array=lambda tensor:tensor.detach().cpu().numpy()
                poses=torch.cat((data.body_link_pos_w-env.scene.env_origins[:,None],data.body_link_quat_w),-1)
                self.trace_recorder.append(public=array(self.last_observation['policy']),
                    episode_ids=array(env.episode_index),residual=array(residual),base=array(base),actual=array(actual),
                    poses=array(poses),ball=array(env.shuttle.data.root_pos_w-env.scene.env_origins),
                    iteration=self.trace_iteration)
            except Exception as error:self.trace_recorder.disable(error)
        observation,reward,dones,extras=self.wrapped.step(actual)
        self.total_control_env_steps+=self.num_envs
        self.last_observation=observation
        ids=torch.where(dones.bool())[0]
        if len(ids):
            self.controller.reset(ids);self.mixer.reset(ids);self._sync_reference()
        if not torch.isfinite(observation['policy']).all() or not torch.isfinite(reward).all():
            raise RuntimeError('Nonfinite native residual rollout')
        if bool(((self.unwrapped.sweep_misses+self.unwrapped.sweep_unresolved)>0).any()):
            raise RuntimeError('Native numerical contact audit failure; no performance accepted')
        record_start=self.last_record_count
        new_records=self.unwrapped.rally_records[record_start:]
        self.last_record_count=len(self.unwrapped.rally_records)
        if any(row['numerical_failures'] for row in new_records):
            raise RuntimeError('Numerical audit failed in an automatically reset native episode')
        if self.trace_recorder is not None and self.trace_recorder.disabled_error is None:
            try:
                env=self.unwrapped
                self.trace_recorder.record_contacts(env.contact_records[self.trace_contact_cursor:])
                self.trace_contact_cursor=len(env.contact_records)
                assert len(env.rally_records)==len(env.episode_records)
                indices=self.trace_recorder.provenance['training_indices'];joined=[]
                for offset,row in enumerate(new_records):
                    jump=env.episode_records[record_start+offset];lane=row['env_id']
                    assert jump['env_id']==lane
                    index=indices[lane%len(indices)]
                    joined.append(dict(row,jump=jump,source_training_index=index,ground_init=True,
                        case_id=f'TRAIN{index}_LANE{lane}_EP{row["episode_index"]}'))
                self.trace_recorder.finish(joined)
            except Exception as error:self.trace_recorder.disable(error)
        return observation,reward,dones,extras


def export_residual_policy(base_checkpoint,policy,joint_scale_rad,public_sample,metadata):
    """Export exactly the actor's public normalization and network, no critic."""
    actor=torch.nn.Sequential(copy.deepcopy(policy.actor_obs_normalizer).cpu().eval(),
                              copy.deepcopy(policy.actor).cpu().eval())
    scripted=torch.jit.script(actor)
    with torch.inference_mode():
        expected=policy.act_inference(TensorDict({'policy':public_sample},batch_size=[len(public_sample)])).cpu()
        # Compare on the actual inference device, avoiding an unnecessary
        # CPU/GPU arithmetic difference in the export fidelity check.
        actual=scripted.to(public_sample.device)(public_sample).cpu()
    maximum_error=float((expected-actual).abs().max())
    if not torch.allclose(actual,expected,atol=2e-5,rtol=2e-5):
        raise RuntimeError('Exported public residual actor disagrees with trained actor')
    blob=io.BytesIO();torch.jit.save(scripted.cpu(),blob);data=blob.getvalue()
    result=copy.deepcopy(base_checkpoint)
    for key in ['training_result','selected_source_run','selected_generation','selected_candidate','selected_training_case_index']:
        result.pop(key,None)
    mode=baseline_mode(metadata.get('base_controller_mode'))
    if mode=='frozen_public_policy':
        validate_frozen_baseline(base_checkpoint)
        for key in ['reactive_trial','native_trial_execution','source_policy_sha256',
                    'source_policy_history_not_inherited','residual_training_metadata']:
            result.pop(key,None)
        result['residual_base_mode']=mode
        result['method']='New bounded whole-body residual over a frozen measured public controller'
    native=bool(metadata.get('native_execution',False))
    result.update(schema=SCHEMA,base_controller=copy.deepcopy(base_checkpoint),residual_actor_jit=data,
        residual_joint_scale_rad=list(joint_scale_rad),native_execution=native,performance_inherited=False,
        formal_acceptance=False,kind=('Actual native residual PPO training; frozen-policy evaluation still required.'
                                    if native else 'CPU residual export check only; no native performance.'),
        residual_training_metadata=dict(metadata,export_actor_max_error=maximum_error,
            export_comparison_device=str(public_sample.device),
            actor_jit_sha256=hashlib.sha256(data).hexdigest(),actor_input='Current public794D observation only; no critic, case IDs or future labels.'))
    return result
