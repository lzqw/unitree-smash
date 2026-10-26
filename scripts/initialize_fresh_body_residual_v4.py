"""Initialize only a new residual; retain the measured two-actor body base."""
import torch


def initialize_fresh_body_actor(policy,configuration):
    if int(policy.actor_obs_normalizer.count)!=0 or int(policy.critic_obs_normalizer.count)!=0:
        raise ValueError('Require fresh learner normalization; source normalizers stay in fixed base')
    std=torch.tensor(configuration['fresh_exploration_std'],device=policy.log_std.device)
    scales=torch.tensor(configuration['joint_scale_rad'],device=std.device)
    if not torch.equal(std.cpu(),torch.tensor([.2]*12+[.1]*6)) or not torch.equal(scales.cpu(),torch.tensor([.15]*18)):
        raise ValueError('Require the explicit requested leg.03/arm.015rad exploration course')
    output=[module for module in policy.actor.modules() if isinstance(module,torch.nn.Linear)][-1]
    if output.out_features!=18:raise ValueError('Require an18D public residual mean')
    preserved={name:value.detach().clone() for name,value in policy.state_dict().items()
        if name.startswith(('critic.','critic_obs_normalizer.','actor_obs_normalizer.'))}
    with torch.no_grad():
        output.weight.zero_();output.bias.zero_();policy.log_std.copy_(std.log())
    assert all(torch.equal(policy.state_dict()[name],value) for name,value in preserved.items())
    return dict(mode='fresh_zero_mean_residual_over_measured_two_actor_body_teacher',
        source_actor_and_normalizers_remain_inside_fixed_base=True,new_actor_mean_initially_zero=True,
        new_actor_normalizer_count=0,new_critic_normalizer_count=0,
        fresh_exploration_std=std.cpu().tolist(),requested_physical_exploration_std_rad=(std*scales).cpu().tolist(),
        actor_transferred=False,critic_transferred=False,critic_normalizer_transferred=False,
        optimizer_transferred=False,old_physical_state_restored=False,source_rng_restored=False,
        previous_learning_counters_restored=False,performance_inherited=False)
