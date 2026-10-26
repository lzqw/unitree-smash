"""Origin0 adapter using the original measured rollout/timeout/trace machinery."""
import copy
import hashlib
import io
import torch
from tensordict import TensorDict
from .full_racket_body_residual_ppo_v4 import (
    ResidualRslAdapterV4 as _MeasuredAdapter, reward_contact_offset,
    ResidualTrainingInterrupted, save_completed_update_error_checkpoint)
from quadsmash.control.full_racket_origin0_residual_v4 import (
    SCHEMA, baseline_mode, validate_frozen_baseline, frozen_origin0_controller, ResidualActionMixerV4)
from quadsmash.control.full_racket_contact_normal_search_v4 import contact_normal_body


class Origin0ResidualAdapterV4(_MeasuredAdapter):
    def __init__(self,wrapped,base_checkpoint,joint_scale_rad,stop_requested=lambda:False,base_mode=None):
        self.wrapped=wrapped;self.base_checkpoint=base_checkpoint
        if baseline_mode(base_mode) != 'frozen_public_policy':
            raise ValueError('Origin0 learning requires an explicit measured fixed source')
        self.controller=frozen_origin0_controller(base_checkpoint,self.num_envs,self.device)
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
    result=dict(parameters=base_checkpoint['parameters'].clone(),asset_sha256=base_checkpoint['asset_sha256'])
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
    result.update(numerical_execution=copy.deepcopy(base_checkpoint['numerical_execution']),
        protocol_sha256=base_checkpoint['protocol_sha256'])
    native=bool(metadata.get('native_execution',False))
    result.update(schema=SCHEMA,base_controller=copy.deepcopy(base_checkpoint),residual_actor_jit=data,
        residual_joint_scale_rad=list(joint_scale_rad),native_execution=native,performance_inherited=False,
        formal_acceptance=False,kind=('Actual native residual PPO training; frozen-policy evaluation still required.'
                                    if native else 'CPU residual export check only; no native performance.'),
        residual_training_metadata=dict(metadata,export_actor_max_error=maximum_error,
            export_comparison_device=str(public_sample.device),
            actor_jit_sha256=hashlib.sha256(data).hexdigest(),actor_input='Current public794D observation only; no critic, case IDs or future labels.'))
    return result
