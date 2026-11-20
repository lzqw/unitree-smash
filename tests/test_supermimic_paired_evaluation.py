import torch

from quadsmash.rl.supermimic_paired_evaluation_v1 import paired_evaluation, parameter_delta


def row(case, contact, ball, recovery, complete, value):
    return dict(case_id=str(case), env_id=case, episode_index=0,
                qualified_airborne_string_contact=contact,
                ball_side_complete_except_recovery=ball,
                recovery_success=recovery, complete5=complete,
                body_ground_seen=not recovery, rally_fault=False,
                numerical_failures=0, native_episode_return=value,
                native_episode_discounted_return=value,
                native_return_definition='undiscounted_sum_of_native_control_step_rewards_v1',
                native_episode_length_steps=800)


def test_paired_evaluation_requires_same_case_identity_and_reports_deltas():
    before=[row(i, i<64, i<48, i<32, i<24, -float(i)) for i in range(128)]
    after=[row(i, i<66, i<50, i<34, i<26, -float(i)+.25) for i in range(128)]
    report=paired_evaluation(before,after,stage='post-Approach',
        actor_before={'head':torch.zeros(2)},actor_after={'head':torch.ones(2)})
    assert report['paired_case_identity_verified']
    assert report['after']['complete5']==26
    assert report['deltas']['complete5_rate'] == .015625
    assert report['actor_parameter_delta']['changed_elements']==2
    assert report['failure_taxonomy_before']['complete5_failure'] == 104

    report=paired_evaluation(before,after,stage='post-Approach',
        action_before={'action':torch.zeros(4,2),'log_std':torch.zeros(2)},
        action_after={'action':torch.ones(4,2),'log_std':torch.zeros(2)})
    assert report['action_rms'] == 1.0 and report['actor_kl'] > 0


def test_parameter_delta_is_explicit_when_unavailable():
    assert parameter_delta(None, {'x': torch.ones(1)}) == {'available': False}
