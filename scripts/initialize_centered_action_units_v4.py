"""Explicit action-unit transfer for a conditional larger-leg TRAIN candidate.

Only the outer mean actor/normalizer transfer. Doubling leg action scales maps
the last affine rows by1/2 and gives fresh leg std0.05,arm std0.1. Physical mean
equivalence is checked on actual source observations. Outside that witness,
old clipped means can differ: this is initialization, never strict resume.
"""
import copy
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from quadsmash.rl.native_checkpoint import interface
from scripts.initialize_centered_policy_v4 import select_centered_public_state, verify_measured_source


def apply_action_unit_mapped_actor(policy,checkpoint,current_interface,target_contract,base,witness):
    old = torch.as_tensor(checkpoint.get('residual_joint_scale_rad',[]),dtype=torch.float32)
    new = torch.as_tensor(target_contract.get('joint_scale_rad',[]),dtype=torch.float32)
    if (old.shape != (18,) or new.shape != (18,) or not torch.equal(old,torch.tensor([.03]*12+[.15]*6))
            or not torch.equal(new,torch.tensor([.06]*12+[.15]*6))):
        raise ValueError('Explicit candidate supports only old leg0.03 to new leg0.06,arm0.15')
    witness = torch.as_tensor(witness,dtype=torch.float32,device='cpu')
    if witness.ndim != 2 or witness.shape[0] == 0 or witness.shape[1] != 794 or not torch.isfinite(witness).all():
        raise ValueError('Require finite actual-source794D observation witnesses')
    if not hasattr(policy,'log_std') or policy.log_std.shape != (18,) or not torch.allclose(
            policy.log_std.detach().cpu().exp(),torch.full((18,),.1),atol=1e-7,rtol=0):
        raise ValueError('Require a fresh log-std actor with configured initial std0.1')
    original = policy.state_dict()
    compatible = copy.deepcopy(target_contract)
    compatible['joint_scale_rad'] = checkpoint['residual_joint_scale_rad']
    mapped,source_actor = select_centered_public_state(checkpoint,current_interface,compatible,base,original)
    with torch.inference_mode():source_mean = source_actor(witness)
    if not torch.isfinite(source_mean).all() or (source_mean[:,:12].abs() > 1.).any():
        raise ValueError('Source leg means clip on the measured witness; action-unit mapping is not equivalent there')
    layers = [(name,module) for name,module in policy.actor.named_modules() if isinstance(module,torch.nn.Linear)]
    if not layers or layers[-1][1].out_features != 18 or list(policy.actor.children())[-1] is not layers[-1][1]:
        raise ValueError('Require an18D final affine mean actor')
    output_prefix = 'actor.'+layers[-1][0]
    ratio = old/new
    mapped[output_prefix+'.weight'] = mapped[output_prefix+'.weight']*ratio[:,None]
    mapped[output_prefix+'.bias'] = mapped[output_prefix+'.bias']*ratio
    preserved = {name:value.detach().clone() for name,value in original.items() if name not in mapped and name != 'log_std'}
    original.update(mapped)
    fresh_std = .1*ratio
    original['log_std'] = fresh_std.log().to(original['log_std'])
    policy.load_state_dict(original,strict=True)
    modes = [(module,module.training) for module in policy.modules()]
    policy.eval()
    device = next(policy.parameters()).device
    with torch.inference_mode():
        actual = policy.actor(policy.actor_obs_normalizer(witness.to(device))).cpu().clamp(-1,1)*new
        expected = source_mean.clamp(-1,1)*old
        error = float((actual-expected).abs().max())
    for module,training in modes:module.training = training
    if not torch.allclose(actual,expected,atol=2e-6,rtol=2e-5):
        raise ValueError('Mapped physical residual means differ on actual-source witness')
    if any(not torch.equal(policy.state_dict()[name],value) for name,value in preserved.items()):
        raise ValueError('Action-unit transfer changed a fresh critic/critic-normalizer tensor')
    return dict(source_joint_scales_rad=old.tolist(),target_joint_scales_rad=new.tolist(),
        output_affine_row_multipliers=ratio.tolist(),actual_witness_rows=len(witness),
        actual_witness_source_leg_mean_max_abs=float(source_mean[:,:12].abs().max()),
        actual_witness_physical_mean_max_error_rad=error,
        source_actor_normalizer_count=int(mapped['actor_obs_normalizer.count']),
        fresh_exploration_std=fresh_std.tolist(),fresh_physical_exploration_std_rad=(fresh_std*new).tolist(),
        exploration_reference='Fresh0.1 in source action units, not the source learned exploration.',
        physical_mean_equivalence_scope='Observed source inputs only. Outside this witness, source-clipped leg means can differ under expanded bounds.',
        actor_and_actor_normalizer_transferred=True,output_action_units_explicitly_changed=True,
        critic_transferred=False,critic_normalizer_transferred=False,optimizer_transferred=False,
        exploration_log_std_transferred=False,source_rng_restored=False,old_physical_state_restored=False,
        previous_learning_counters_restored=False,performance_inherited=False)


def initialize_centered_action_units(runner,path,evaluation,base,root):
    if (runner.current_learning_iteration != 0 or runner.tot_timesteps != 0 or runner.tot_time != 0
            or runner.alg.optimizer.state or getattr(runner,'_learning_now',False)):
        raise ValueError('Action-unit transfer requires fresh learner/Adam/counters')
    state = runner.alg.policy.state_dict()
    if any(int(state[name]) != 0 for name in ['actor_obs_normalizer.count','critic_obs_normalizer.count']):
        raise ValueError('Action-unit transfer requires fresh normalization')
    contract = runner.original_training_config['residual_contract']
    provenance = verify_measured_source(root,path,evaluation,contract)
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != provenance['source_policy_sha256']:
        raise ValueError('Measured source changed during transfer')
    checkpoint = torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    witness_path = Path(evaluation)/'public_and_actions_first2.npz'
    witness_bytes = witness_path.read_bytes()
    with np.load(io.BytesIO(witness_bytes),allow_pickle=False) as arrays:
        observations,actions,valid = arrays['observations'],arrays['actions'],arrays['valid_first_episode']
        if observations.shape != (800,2,794) or actions.shape != (800,2,18) or valid.shape != (800,2):
            raise ValueError('Require the actual original first2 native observation/action record')
        witness = torch.from_numpy(observations[valid.astype(bool)].copy())
    result = apply_action_unit_mapped_actor(runner.alg.policy,checkpoint,interface(runner.env.unwrapped),contract,base,witness)
    runner.env.reset()
    return dict(mode='fresh_measured_centered_actor_with_explicit_action_unit_rescaling',
        num_envs=runner.env.num_envs,initial_iteration=0,current_job_steps_start_at_zero=True,
        source_completed_updates=checkpoint['residual_training_metadata']['completed_updates'],
        source_public_policy_is_not_a_restart_checkpoint=True,
        actual_source_observation_witness=str(witness_path),
        actual_source_observation_witness_sha256=hashlib.sha256(witness_bytes).hexdigest(),
        **provenance,**result)
