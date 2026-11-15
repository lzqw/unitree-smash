import torch

from quadsmash.rl.supermimic_ball_side_shadow_v1 import ball_side_shadow_row, summarize_ball_side_shadow
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer


class _State:
    observation = torch.zeros(2, 794)


def test_ball_side_shadow_never_turns_missing_evidence_into_success():
    row = {'case_id': 'T17', 'episode_index': 0, 'ball_side_complete_except_recovery': True}
    result = ball_side_shadow_row(row)
    assert result['shadow_ball_side_ok'] is None
    assert result['enforce'] is False
    assert 'outgoing_velocity' in result['missing_fields']


def test_ball_side_shadow_reports_observed_rows_only():
    row = {name: False for name in (
        'qualified_airborne_string_contact', 'ball_side_complete_except_recovery',
        'crossed_net', 'landed_legal_court', 'landed_target', 'rally_fault')}
    row.update(case_id='T1', episode_index=0, outgoing_velocity=[1, 0, 0],
               contact_point=[0, 0, 0], contact_time=1.0, net_margin=0.1)
    row['ball_side_complete_except_recovery'] = True
    result = summarize_ball_side_shadow([row])
    assert result['mode'] == 'shadow'
    assert result['complete_observed'] == 1
    assert result['missing_evidence_rows'] == 0


def test_stage_rtg_audit_keeps_policy_episode_and_time_labels():
    rows = PhysicalHorizonTrainer.audit_native_rtg_stages(
        [_State(), _State()], [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        [dict(policy_id='pi0', episode_id='ep0', time_s=0.0),
         dict(policy_id='pi0', episode_id='ep0', time_s=3.84)],
    )
    assert rows['critic_updates'] == 0
    assert rows['total_labels'] == 4
    assert [x['time_s'] for x in rows['stages']] == [0.0, 3.84]


def test_budget_delta_uses_final_vector_shapes():
    now = (torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0]), torch.tensor([0.5, 0.8]))
    base = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 3.0]), torch.tensor([0.2, 0.3]))
    assert PhysicalHorizonTrainer._max_budget_delta(now, base) == (2.0, 1.0, 0.5)
