import pytest
from quadsmash.rl.supermimic_frozen_return_monitor_v1 import FrozenReturnMonitor


def records(value,policy='p'):
    return [dict(env_id=i,case_id=str(i),policy_id=policy,actor_sha256=policy,native_episode_return=value,
                 native_return_definition='undiscounted_sum_of_native_control_step_rewards_v1',native_episode_length_steps=800) for i in range(128)]


def observe(m,value,policy='p'):
    return m.observe(records(value,policy),policy_id=policy,actor_sha256=policy,contract_sha256='fixed-task')


def test_decline_stop_survives_resume_and_never_promotes():
    m=FrozenReturnMonitor();assert observe(m,-10)['stop_reason'] is None
    observe(m,-11,'p1')
    restored=FrozenReturnMonitor();restored.load_state_dict(m.state_dict())
    result=observe(restored,-12,'p2')
    assert result['stop_reason']=='native-frozen-return-declining' and not result['formal_promotion']
    with pytest.raises(RuntimeError):observe(restored,-9,'p3')


def test_plateau_is_explicitly_distinct_from_proven_convergence():
    m=FrozenReturnMonitor();observe(m,-10);observe(m,-9.99,'p1')
    result=observe(m,-9.98,'p2')
    assert result['stop_reason']=='native-frozen-return-plateau'
    assert 'not proof' in result['scope']


def test_partial_mixed_and_imagined_returns_rejected():
    for rows in (records(-10)[:2],records(-10)):
        m=FrozenReturnMonitor()
        if len(rows)==128:rows[0]['mixed_policy']=True
        with pytest.raises(ValueError):m.observe(rows,policy_id='p',actor_sha256='p',contract_sha256='fixed-task')
    rows=records(-10);rows[0]['native_return_definition']='imagined'
    with pytest.raises(ValueError):FrozenReturnMonitor().observe(rows,policy_id='p',actor_sha256='p',contract_sha256='fixed-task')


def test_resume_threshold_changes_and_case_contract_changes_rejected():
    m=FrozenReturnMonitor();observe(m,-10)
    with pytest.raises(ValueError):FrozenReturnMonitor(patience=3).load_state_dict(m.state_dict())
    with pytest.raises(ValueError):m.observe(records(-9),policy_id='p',actor_sha256='p',contract_sha256='changed-task')
