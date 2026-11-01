"""Fresh residual initialization using explicitly configured exploration.

Only the residual output layer and its Gaussian scale are initialized here.
The shared teacher remains separate, and no previous learner is inherited.
"""
import torch


def initialize_shared_target_actor(policy, configuration):
    if int(policy.actor_obs_normalizer.count) or int(policy.critic_obs_normalizer.count):
        raise ValueError('Require a new learner with fresh observation normalization')
    std = torch.as_tensor(configuration['fresh_exploration_std'], device=policy.log_std.device)
    scales = torch.as_tensor(configuration['joint_scale_rad'], device=std.device)
    if (std.shape != (18,) or scales.shape != (18,)
            or not torch.isfinite(std).all() or not torch.isfinite(scales).all()
            or (std <= 0).any() or (scales <= 0).any()):
        raise ValueError('Declare 18 positive finite exploration and joint scales')
    output = [layer for layer in policy.actor.modules() if isinstance(layer, torch.nn.Linear)][-1]
    if output.out_features != 18 or policy.log_std.shape != (18,):
        raise ValueError('Require the 18D residual interface')
    with torch.no_grad():
        output.weight.zero_()
        output.bias.zero_()
        policy.log_std.copy_(std.log())
    result = dict(mode='fresh_zero_mean_shared_target_residual',
        new_actor_mean_initially_zero=True,
        fresh_exploration_std=std.cpu().tolist(),
        physical_exploration_std_rad=(std*scales).cpu().tolist(),
        actor_transferred=False, critic_transferred=False, optimizer_transferred=False,
        source_rng_restored=False, old_physical_state_restored=False,
        previous_learning_counters_restored=False, performance_inherited=False)
    if hasattr(policy, 'landing_exploration_scale'):
        result.update(physical_exploration_std_scope='Base sampling std outside observed recovery phase',
            recovery_exploration_multiplier=policy.landing_exploration_scale,
            recovery_physical_exploration_std_rad=(std*scales*policy.landing_exploration_scale).cpu().tolist())
    return result
