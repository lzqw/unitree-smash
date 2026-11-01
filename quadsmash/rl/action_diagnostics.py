"""CPU-testable, RNG-free diagnostics for the locked diagonal Gaussian policy."""
import math
import torch


def policy_std(policy):
    if getattr(policy, 'noise_std_type', None) == 'log':
        return policy.log_std.exp()
    if getattr(policy, 'noise_std_type', None) == 'scalar':
        return policy.std
    raise ValueError('unsupported policy noise representation')


@torch.no_grad()
def reset_initial_std(policy, desired_std):
    """Preserve the existing init-only reset, adding an effective-value assertion."""
    if not math.isfinite(desired_std) or desired_std <= 0:
        raise ValueError('initial noise std must be finite and positive')
    before = policy_std(policy).detach().clone()
    if policy.noise_std_type == 'log':
        policy.log_std.fill_(math.log(desired_std))
    else:
        policy.std.fill_(desired_std)
    actual = policy_std(policy)
    if not torch.isfinite(actual).all() or not torch.allclose(actual, torch.full_like(actual, desired_std), rtol=1e-6, atol=1e-7):
        raise RuntimeError('loaded policy noise reset did not take effect')
    return {'requested_std': desired_std, 'before_per_joint': before.cpu().tolist(),
            'effective_std_per_joint': actual.cpu().tolist(),
            'std_remains_learnable': bool((policy.log_std if policy.noise_std_type == 'log' else policy.std).requires_grad)}


@torch.no_grad()
def action_summary(mean, std, raw_actions=None):
    """No action sampling; clipping is the real env's [-1,1] command boundary.

    Input rows may be an explicitly documented rollout subset. This does not
    claim to measure joint target limits or actuator saturation.
    """
    std = torch.broadcast_to(std, mean.shape)
    if mean.ndim != 2 or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('expected finite [samples,joints] mean and positive std')
    low, high = (-1. - mean) / std, (1. - mean) / std
    cdf_low, cdf_high = torch.special.ndtr(low), torch.special.ndtr(high)
    density = lambda x: torch.exp(-.5 * x.square()) / math.sqrt(2. * math.pi)
    expected = -cdf_low + mean * (cdf_high - cdf_low) + std * (density(low) - density(high)) + 1. - cdf_high
    result = {'samples': len(mean), 'mean_outside_bounds_fraction': float((mean.abs() > 1.).float().mean()),
        'mean_outside_bounds_per_joint': (mean.abs() > 1.).float().mean(0).cpu().tolist(),
        'mean_abs_per_joint': mean.abs().mean(0).cpu().tolist(),
        'std_per_joint': std.mean(0).cpu().tolist(),
        'std_min_per_joint': std.amin(0).cpu().tolist(), 'std_max_per_joint': std.amax(0).cpu().tolist(),
        'expected_clipping_fraction': float((cdf_low + 1. - cdf_high).mean()),
        'expected_clipping_per_joint': (cdf_low + 1. - cdf_high).mean(0).cpu().tolist(),
        'clipped_mean_vs_expected_action_gap_per_joint': (mean.clamp(-1., 1.) - expected).abs().mean(0).cpu().tolist()}
    if raw_actions is not None:
        if raw_actions.shape != mean.shape or not torch.isfinite(raw_actions).all():
            raise ValueError('raw actions must be finite and align with mean rows')
        result.update(actual_clipped_fraction=float((raw_actions.abs() > 1.).float().mean()),
            actual_clipped_fraction_per_joint=(raw_actions.abs() > 1.).float().mean(0).cpu().tolist(),
            raw_to_applied_l1_per_joint=(raw_actions - raw_actions.clamp(-1., 1.)).abs().mean(0).cpu().tolist())
    return result


@torch.no_grad()
def post_update_diagnostics(policy, obs, old_mean, old_std, actions, old_logprob, clip_param):
    # Locked RSL-RL act_inference only forwards the MLP/normalizer. It does not
    # call update_normalization, sample, or replace policy.distribution.
    mean = policy.act_inference(obs)
    conditional_std = getattr(policy, 'action_std_for_observations', None)
    std = conditional_std(obs) if conditional_std is not None else policy_std(policy).expand_as(mean)
    kl = (torch.log(std / old_std) + (old_std.square() + (old_mean - mean).square()) / (2 * std.square()) - .5).sum(-1)
    logprob = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
    ratio = torch.exp(logprob - old_logprob.squeeze(-1))
    return {'post_update_kl': float(kl.mean()),
            'post_update_clip_fraction': float(((ratio - 1.).abs() > clip_param).float().mean()),
            'action_std_mean': float(std.mean()),
            'behavior_actions': action_summary(old_mean, old_std, actions),
            'post_update_actions': action_summary(mean, std)}
