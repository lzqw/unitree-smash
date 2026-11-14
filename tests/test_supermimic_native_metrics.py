import copy
import pytest
import torch
from quadsmash.rl.supermimic_native_metrics import (
    NativeReturnAccumulator, NativeEpisodeStatistics, summarize_native_episodes, RETURN_DEFINITION)


def scored(contact, recovery=False, value=None, length=2):
    row=dict(qualified_airborne_string_contact=contact,recovery_success=recovery,
             ball_side_complete_except_recovery=contact,complete5=contact and recovery)
    if value is not None:
        row.update(native_episode_return=value,native_episode_length_steps=length,native_return_definition=RETURN_DEFINITION)
    return row


def test_terminal_reward_included_and_async_resets_do_not_mix_returns():
    a=NativeReturnAccumulator(2,'cpu')
    a.add(torch.tensor([1.,10.],requires_grad=True));a.add(torch.tensor([2.,20.]))
    first=a.records([0])[0]
    assert first['native_episode_return']==3 and first['native_episode_length_steps']==2
    a.reset(torch.tensor([0]));a.add(torch.tensor([-4.,30.]))
    now=a.records([0,1])
    assert now[0]['native_episode_return']==-4 and now[0]['native_episode_length_steps']==1
    assert now[1]['native_episode_return']==60 and now[1]['native_episode_length_steps']==3
    assert first['native_episode_return']==3
    assert not a.returns.requires_grad


def test_dynamic_denominator_and_unqualified_recovery_not_counted():
    # This is intentionally not a 74-lane fixture.
    rows=[scored(True,True,5),scored(True,False,-1),scored(False,True,8)]
    before=copy.deepcopy(rows);result=summarize_native_episodes(rows)
    assert result['conditional_recovery_denominator']==2
    assert result['conditional_recovery_count']==1
    assert result['recovery_given_airborne_string_contact']==.5
    assert result['qualified_airborne_string_contact_rate']==2/3
    assert result['complete5_rate']==1/3 and rows==before
    assert result['native_episode_return_mean']==4
    rows.append(scored(True,True,0))
    assert summarize_native_episodes(rows)['recovery_given_airborne_string_contact']==2/3


def test_no_completed_episode_or_no_contact_is_unknown_not_zero_success():
    empty=NativeEpisodeStatistics().snapshot()
    assert empty['completed_episodes']==0
    assert empty['native_episode_return_mean'] is None
    assert empty['recovery_given_airborne_string_contact'] is None
    result=summarize_native_episodes([scored(False,False)])
    assert result['native_episode_return_missing_count']==1
    assert result['native_episode_return_mean'] is None
    assert result['conditional_recovery_denominator']==0
    assert result['recovery_given_airborne_string_contact'] is None


def test_return_is_raw_sum_without_second_dt_or_discount():
    a=NativeReturnAccumulator(1,'cpu')
    for r in (.005,20.,-40.):a.add(torch.tensor([r],dtype=torch.float64))
    record=a.records([0])[0]
    assert record['native_episode_return']==pytest.approx(-19.995)
    assert record['native_return_definition']==RETURN_DEFINITION
    with pytest.raises(ValueError,match='lane shape'):a.add(torch.zeros(2))


def test_invalid_return_is_not_silently_reported_as_zero():
    with pytest.raises(ValueError,match='Invalid native'):summarize_native_episodes([scored(True,value=float('nan'))])
    with pytest.raises(ValueError,match='Invalid native'):summarize_native_episodes([scored(True,value=5,length=0)])


def test_missing_task_evidence_is_not_treated_as_false():
    row=scored(True)
    row['recovery_success']=None
    with pytest.raises(ValueError,match='Missing measured'):
        summarize_native_episodes([row])
