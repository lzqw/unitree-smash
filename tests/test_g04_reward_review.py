import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.audit_g04_rewards_cpu import signal_probe, PROTOCOL
from quadsmash.tasks.stand.jump_reward_audit import v2c_components, recovery_draft, RecoveryWindows
from quadsmash.tasks.stand import jump_metrics


@pytest.mark.parametrize('kind', ['tiny_hop', 'short_high_hop', 'long_low_hop'])
def test_unqualified_hops_earn_old_main_recovery_but_not_draft(kind):
    result = signal_probe(kind)
    assert result['recovery_seen'] and not result['valid_jump_landed'] and not result['formal_success']
    assert result['old_primary_recovery'] == pytest.approx(5., abs=1e-5)
    assert result['draft_primary_recovery'] == 0.
    assert 0 < result['draft_weak_recovery'] <= .1


def test_qualified_jump_keeps_task_credit_and_formal_success():
    result = signal_probe('qualified')
    assert result['valid_jump_landed'] and result['formal_success']
    assert result['draft_primary_recovery'] == pytest.approx(result['old_primary_recovery'])


@pytest.mark.parametrize('kind', ['body_contact', 'standing'])
def test_standing_and_unsafe_landings_cannot_earn_recovery(kind):
    result = signal_probe(kind)
    assert not result['formal_success']
    assert result['draft_primary_recovery'] == result['draft_weak_recovery'] == 0.


def test_broken_recovery_keeps_shaping_but_never_passes_acceptance():
    result = signal_probe('broken_recovery')
    assert result['valid_jump_landed'] and not result['formal_success']
    assert result['max_continuous_recovery_s'] < .31
    assert result['draft_primary_recovery'] > 0.


def test_weak_budget_is_per_episode_and_cannot_be_replenished_by_hops():
    paid = torch.zeros(2); mask = torch.zeros(2, dtype=torch.bool)
    components = {'stable_recovery': torch.ones(2), 'recovery_quality': torch.ones(2), 'other': torch.tensor([2., 3.])}
    initial = {k: v.clone() for k, v in components.items()}
    accumulated = torch.zeros(2)
    for i in range(1000):
        quality = torch.ones(2) if i % 100 < 90 else torch.zeros(2)
        draft, paid = recovery_draft(components, valid_jump_landed=mask, body_ground_seen=mask,
            recovery_quality=quality, weak_paid=paid, dt=.01)
        accumulated += draft['weak_safe_settling']
        assert torch.equal(draft['other'], components['other'])
    assert torch.allclose(accumulated, torch.full((2,), .1), atol=1e-6)
    assert all(torch.equal(components[k], v) for k, v in initial.items())
    paid[0] = 0.  # a real partial episode reset, not a new hop
    draft, _ = recovery_draft(components, valid_jump_landed=mask, body_ground_seen=mask,
        recovery_quality=torch.ones(2), weak_paid=paid, dt=.01)
    assert draft['weak_safe_settling'][0] > 0 and draft['weak_safe_settling'][1] == 0


def _method(path):
    module = ast.parse(Path(path).read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'JumpEnv')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_get_rewards')
    ns = {'torch': torch}; exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    return ns['_get_rewards']


def _fake_env():
    n = 24; g = torch.Generator().manual_seed(311003)
    rand = lambda *shape: torch.rand(*shape, generator=g)
    boolean = lambda: rand(n) > .5
    data = SimpleNamespace(joint_pos=rand(n, 18), default_joint_pos=rand(n, 18),
        root_ang_vel_w=rand(n, 3), joint_vel=rand(n, 18), projected_gravity_b=rand(n, 3),
        root_link_lin_vel_w=rand(n, 3), applied_torque=rand(n, 18))
    e = SimpleNamespace(robot=SimpleNamespace(data=data), _episode_has_stepped=boolean(),
        _phase=lambda: torch.linspace(0, 4., n), reset_com_height=rand(n), com_position=rand(n, 3),
        clear_flight=boolean(), flight_apex_rise=rand(n), stable_recovery=boolean(),
        flight_qualified=boolean(), rewarded_jump=boolean(), jump_success=boolean(), rewarded_success=boolean(),
        body_ground_seen=boolean(), rewarded_body_contact=boolean(), joint_ids=list(range(18)),
        landed_once=boolean(), observed_foot_contact=rand(n, 4)>.5, action_delta=rand(n),
        step_dt=.01, reset_terminated=boolean())
    return e


def test_default_reward_is_bitwise_identical_to_reviewed_commit_with_or_without_observer():
    old = _method('tests/fixtures/jump_env_65396a21.py.txt')
    current = _method('quadsmash/tasks/stand/jump_env.py')
    baseline = _fake_env(); off = copy.deepcopy(baseline); on = copy.deepcopy(baseline)
    recorded = []
    on._diagnostic_observer = SimpleNamespace(on_reward=lambda env, f, reward: recorded.append((v2c_components(f, env.step_dt), reward.clone())))
    expected = old(baseline)
    assert torch.equal(expected, current(off))
    assert torch.equal(expected, current(on))
    assert torch.allclose(sum(recorded[0][0].values()), expected, atol=2e-6, rtol=0.)
    for name in ('rewarded_jump', 'rewarded_success', 'rewarded_body_contact', '_episode_has_stepped'):
        assert torch.equal(getattr(baseline, name), getattr(on, name))


def test_recovery_condition_windows_break_individually_and_partial_reset_isolated():
    state = SimpleNamespace(); jump_metrics.initialize(state, 2, 'cpu')
    state.landed_once[:] = True; state.valid_jump_landed[:] = True
    monitor = RecoveryWindows(2, 'cpu')
    saved = {k: v.clone() for k, v in vars(state).items()}
    def tick(speed):
        monitor.observe(state, dt=.01, support=torch.ones(2, dtype=torch.bool), roll_pitch=torch.zeros(2, 2),
            horizontal_speed=torch.zeros(2), angular_speed=torch.zeros(2), joint_speed=torch.tensor(speed),
            com_vertical_velocity=torch.zeros(2), protocol=PROTOCOL)
    for _ in range(10): tick([0., 0.])
    tick([2., 0.])
    j = monitor.names.index('joint_speed'); a = monitor.names.index('all')
    assert monitor.run[0, j] == monitor.run[0, a] == 0.
    assert monitor.maximum[0, j] == pytest.approx(.1)
    assert monitor.run[1, j] == pytest.approx(.11)
    assert all(torch.equal(v, getattr(state, k)) for k, v in saved.items())
    other = monitor.run[1].clone(); monitor.reset(torch.tensor([0]))
    assert monitor.run[0].sum() == 0 and torch.equal(other, monitor.run[1])
