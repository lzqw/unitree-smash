"""Transfer a measured outer actor on the same fixed base into a fresh PPO.

The source is a public inference policy, so an independently measured native
intermediate actor is eligible without pretending its diagnostic is canonical.
Only outer actor/normalizer tensors transfer; critic, exploration, Adam, RNG,
physical state and experiment counters belong to the new learner.
"""
import hashlib
import io
import json
import math
import numbers
from pathlib import Path

import numpy as np
import torch

from quadsmash.control.full_racket_residual_v4 import SCHEMA, validate_frozen_baseline
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, summarize_training
from quadsmash.rl.native_checkpoint import interface


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def same_tree(left,right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.shape == right.shape
                and left.dtype == right.dtype and torch.equal(left,right))
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return (isinstance(left,np.ndarray) and isinstance(right,np.ndarray) and left.dtype == right.dtype
                and np.array_equal(left,right,equal_nan=True))
    if isinstance(left,dict) or isinstance(right,dict):
        return (isinstance(left,dict) and isinstance(right,dict) and left.keys() == right.keys()
                and all(same_tree(value,right[key]) for key,value in left.items()))
    if isinstance(left,(list,tuple)) or isinstance(right,(list,tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(same_tree(a,b) for a,b in zip(left,right)))
    if isinstance(left,numbers.Real) and isinstance(right,numbers.Real):
        return bool(left == right or (math.isnan(left) and math.isnan(right)))
    return type(left) is type(right) and left == right


def select_centered_public_state(checkpoint,current_interface,target_contract,base,current_model):
    if (checkpoint.get('schema') != SCHEMA or not checkpoint.get('native_execution')
            or checkpoint.get('residual_base_mode') != 'frozen_public_policy'):
        raise ValueError('Use an actual native outer residual over one fixed learned base')
    if (target_contract.get('base_controller_mode') != 'frozen_public_policy'
            or current_interface.get('action_family') != 'bounded-residual-joint-targets-full-racket-v4'
            or current_interface.get('observation_size') != 794
            or current_interface.get('history_length') != 11 or current_interface.get('policy_dt') != .01
            or current_interface.get('asset_sha256') != checkpoint.get('asset_sha256')):
        raise ValueError('Centered transfer must preserve the public full-racket interface')
    validate_frozen_baseline(base)
    if not same_tree(checkpoint.get('base_controller'),base):
        raise ValueError('The fixed base changed; do not add a learned actor twice')
    if (checkpoint.get('residual_joint_scale_rad') != target_contract.get('joint_scale_rad')
            or not torch.equal(checkpoint['parameters'],base['parameters'])):
        raise ValueError('Public outer-actor action units or analytic parameters changed')
    metadata = checkpoint.get('residual_training_metadata',{})
    blob = checkpoint.get('residual_actor_jit')
    if (not isinstance(blob,bytes) or hashlib.sha256(blob).hexdigest() != metadata.get('actor_jit_sha256')
            or metadata.get('base_controller_mode') != 'frozen_public_policy'
            or metadata.get('base_policy_sha256') != target_contract.get('base_policy_sha256')
            or metadata.get('actual_control_env_steps',0) <= 0 or metadata.get('completed_updates',0) <= 0):
        raise ValueError('Require matching actual native outer-actor export provenance')
    actor = torch.jit.load(io.BytesIO(blob),map_location='cpu').eval()
    source = actor.state_dict()
    if any(not torch.isfinite(value).all() for value in source.values()):
        raise ValueError('Nonfinite source outer actor or normalizer')
    mapped = {}
    for key,value in source.items():
        if key.startswith('0.'):name = 'actor_obs_normalizer.'+key[2:]
        elif key.startswith('1.'):name = 'actor.'+key[2:]
        else:raise ValueError('Unexpected public actor export state: '+key)
        mapped[name] = value
    expected = {name for name in current_model if name.startswith(('actor.','actor_obs_normalizer.'))}
    if (set(mapped) != expected or not any(name.startswith('actor.') for name in expected)
            or 'actor_obs_normalizer.count' not in expected):
        raise ValueError('Public actor architecture or normalizer is incompatible')
    for name,value in mapped.items():
        target = current_model[name]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError('Outer actor architecture mismatch: '+name)
    return mapped,actor


def verify_measured_source(root,policy_path,evaluation,target_contract):
    root,policy_path,evaluation = Path(root),Path(policy_path),Path(evaluation)
    policy_sha = digest(policy_path)
    inputs,metrics,job = [read(evaluation/name) for name in ['INPUTS.json','metrics.json','job.json']]
    protocol_path = root/'configs/g07_full_racket_eval_v4_r3.json'
    protocol = read(protocol_path); protocol['_root'] = str(root)
    if (target_contract.get('protocol_sha256') != digest(protocol_path)
            or inputs.get('protocol_sha256') != digest(protocol_path)
            or metrics.get('protocol_sha256') != digest(protocol_path)
            or inputs.get('policy_sha256') != policy_sha or metrics.get('policy_sha256') != policy_sha
            or inputs.get('formal_cases_read') is not False):
        raise ValueError('Source policy lacks a matching original TRAIN evaluation')
    if (job.get('status') != 'SUCCEEDED' or job.get('exit_code') != 0
            or not job.get('receipt_present') or job.get('charged_env_steps') != 102400
            or read(evaluation/'completed.json').get('env_steps') != 102400):
        raise ValueError('Source requires successful actual native evaluation receipts')
    amplitudes = inputs.get('reset_noise_amplitudes',{})
    if (inputs.get('training_indices') != list(range(128))
            or set(amplitudes) != {'joint_position_rad','joint_velocity_rad_s','root_xy_m','root_yaw_rad'}
            or any(value != 0. for value in amplitudes.values())
            or metrics.get('noise_level') != 'none'):
        raise ValueError('Use a complete original128-case zero-noise TRAIN source, not formal feedback')
    source_hashes = read(evaluation/'source_hashes.json')
    # This path adapts TRAIN rewards while preserving the observed dynamics,
    # public observations and the executable baseline/controller semantics.
    for name,expected in source_hashes.items():
        if name.startswith('quadsmash/') and digest(root/name) != expected:
            raise ValueError('Source/target runtime changed: '+name)
    for name,expected in protocol['physical_configs'].items():
        if digest(root/name) != expected:raise ValueError('Original physical config changed: '+name)
    if digest(root/protocol['asset']['path']) != protocol['asset']['sha256']:
        raise ValueError('Original full-racket asset changed')
    cases,indices = training_cases(root,protocol)
    rows = read(evaluation/'episodes.json')
    if (len(rows) != 128 or [row['source_training_index'] for row in rows] != indices
            or any(row['env_id'] != index or row['episode_index'] != 0 or not row['ground_init'] for index,row in enumerate(rows))):
        raise ValueError('Source assessment omits original first ground-start cases')
    summary = summarize_training(rows,cases,indices,protocol)
    if summary['numerical_failures'] != 0 or any(metrics[key] != value for key,value in summary.items()):
        raise ValueError('Source native/independent TRAIN results disagree')
    return dict(source_policy_sha256=policy_sha,source_evaluation=str(evaluation),
        source_original_TRAIN_summary=summary,source_evidence_sha256={name:digest(evaluation/name) for name in
            ['INPUTS.json','metrics.json','job.json','completed.json','episodes.json','source_hashes.json']},
        source_protocol_sha256=digest(protocol_path),source_performance_is_not_inherited=True)


def apply_centered_public_actor(policy,checkpoint,current_interface,target_contract,base):
    """Pure model transfer; no environment, optimizer or experiment state API."""
    original = policy.state_dict()
    mapped,source_actor = select_centered_public_state(checkpoint,current_interface,target_contract,base,original)
    preserved = {name:value.detach().clone() for name,value in original.items() if name not in mapped}
    original.update(mapped);policy.load_state_dict(original,strict=True)
    # Check without updating either public normalization or the fresh critic.
    modes = [(module,module.training) for module in policy.modules()]
    policy.eval()
    probe = torch.linspace(-.25,.25,794,device=next(policy.parameters()).device).repeat(4,1)
    with torch.inference_mode():
        expected = source_actor(probe.cpu()).to(probe.device)
        actual = policy.actor(policy.actor_obs_normalizer(probe))
        error = float((actual-expected).abs().max())
    for module,training in modes:module.training = training
    if not torch.allclose(actual,expected,atol=2e-5,rtol=2e-5):
        raise ValueError('Transferred mean actor differs from the measured public actor')
    if any(not torch.equal(policy.state_dict()[name],value) for name,value in preserved.items()):
        raise ValueError('Transfer changed a fresh critic or exploration tensor')
    return dict(actor_export_max_error=error,source_actor_normalizer_count=int(mapped['actor_obs_normalizer.count']),
        public_actor_and_actor_normalizer_transferred=True,exploration_log_std_transferred=False,
        critic_transferred=False,critic_normalizer_transferred=False,optimizer_transferred=False,
        source_rng_restored=False,old_physical_state_restored=False,previous_learning_counters_restored=False,
        one_fixed_base_and_one_outer_residual=True,performance_inherited=False)


def initialize_centered_policy(runner,path,evaluation,base,root):
    if (runner.current_learning_iteration != 0 or runner.tot_timesteps != 0 or runner.tot_time != 0
            or runner.alg.optimizer.state or getattr(runner,'_learning_now',False)):
        raise ValueError('Centered actor transfer requires fresh learner/Adam/counters')
    current = runner.alg.policy.state_dict()
    if any(int(current[name]) != 0 for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']):
        raise ValueError('Centered actor transfer requires fresh public/critic normalization')
    contract = runner.original_training_config['residual_contract']
    provenance = verify_measured_source(root,path,evaluation,contract)
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != provenance['source_policy_sha256']:
        raise ValueError('Measured source policy changed during transfer verification')
    checkpoint = torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    result = apply_centered_public_actor(runner.alg.policy,checkpoint,interface(runner.env.unwrapped),contract,base)
    runner.env.reset()
    return dict(mode='fresh_learner_with_measured_centered_public_actor',num_envs=runner.env.num_envs,
        initial_iteration=0,current_job_steps_start_at_zero=True,source_completed_updates=checkpoint['residual_training_metadata']['completed_updates'],
        source_public_policy_is_not_a_restart_checkpoint=True,**provenance,**result)
