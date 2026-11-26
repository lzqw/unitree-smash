import hashlib
import json
from pathlib import Path

import pytest
import torch

from quadsmash.rl.supermimic_approach_data_v1 import approach_guard_merit
from quadsmash.rl.supermimic_approach_speed_risk import (
    compare_risk, forward_tolerances, group_mean, incomplete_risk, release_risk, rejudge_history,
)


def test_grouped_speed_losses_keep_gradient_for_fast_and_incomplete_risk_lanes():
    speed = torch.tensor([5.6, 6.5], requires_grad=True)
    geometry = torch.tensor([.4], requires_grad=True)
    risky = torch.cat([release_risk(speed, torch.zeros_like(speed)), incomplete_risk(geometry)])
    groups = torch.tensor([1, 2, 2])
    loss = group_mean(risky, groups)
    full = group_mean(torch.cat([risky, torch.ones(29)]), torch.cat([groups, torch.zeros(29, dtype=torch.long)]))
    torch.testing.assert_close(full, loss)
    loss.backward()
    assert (speed.grad < 0).all()
    assert geometry.grad.item() > 0


def test_frozen_guard_uses_measured_tolerance_and_preserves_marginal_release():
    def rollout(flight, risk, release, loss):
        return dict(flight_cost_by_lane=torch.tensor(flight), string_margin_by_lane=torch.tensor([.03, .03]),
                    speed_risk_by_lane=(torch.tensor(risk), torch.tensor([1, 2]), torch.tensor(release)), loss=loss)

    baseline = rollout([.1, .1], [.5, .5], [True, True], 1.)
    tolerance = forward_tolerances(baseline, baseline)
    assert tolerance['risk'] < 1e-6 and tolerance['flight'] < 1e-6
    candidate = rollout([.098, .099], [.501, .499], [True, True], .9)
    assert approach_guard_merit(baseline, candidate, flight_tolerance=tolerance['flight'])[0]
    assert not compare_risk(baseline, candidate, tolerance['risk'])[0]
    lost_release = rollout([.04, float('nan')], [.4, 2.2], [True, False], .9)
    ok, details = compare_risk(baseline, lost_release, tolerance['risk'])
    assert not ok and details['marginal_success']['lost_releases'] == 1


def test_historical_rejudge_preserves_missing_evidence_and_two_compensated_trials():
    root = Path(__file__).resolve().parents[1]
    history = json.loads((root / 'reports/supermimic_approach_ball_only_20261123/E1_V2_A40_SHORT_REFRESH_SPEED_RISK.json').read_text())
    audit = rejudge_history(history, {'flight': 1e-6, 'risk': 1e-6})
    assert audit['total_logged'] == 11
    rows = audit['accepted_trials']
    assert len(rows) == 8
    assert sum(row['flight_failed'] for row in rows) == 6
    assert sum(not row['flight_failed'] and row['risk_groups']['low_speed']['comparable_failed'] for row in rows) == 2
    assert sum(row['risk_groups']['low_speed']['incomplete'] for row in rows) == 3


def test_pooled_objective_normalizes_risk_across_start_records():
    from types import SimpleNamespace
    from quadsmash.rl.supermimic_recovery_batches import pooled_objective

    actor = torch.nn.Linear(1, 1, bias=False)
    weight = actor.weight.reshape(())
    trainer = SimpleNamespace(actor=actor, block_name='approach', approach_objective='toi_racket_only',
                              approach_v2=True, approach_speed_risk=True, approach_ball_protocol={},
                              tracking_coefficient=lambda _: 1., anchor_coeff=0.)

    def rollout(start, *, effective_weights=None, baseline_toi_weights=None, collect_critic_data=False,
                defer_speed_risk=False):
        assert defer_speed_risk
        values, groups = start
        n = len(values)
        active = torch.ones(n, dtype=torch.bool)
        zero = weight * 0.
        return dict(task=zero, tracking=zero, anchor=zero, effective_weights=active, ret=torch.zeros(n),
                    toi_weights=None, toi_coverage_preserved=True, speed_risk_by_lane=(values * weight, groups, active),
                    guard_flight_cost_by_lane=values * weight,
                    reference_coverage=dict(selected_start_lanes=n, gate_active_selected_start_lanes=n),
                    gate_lanes=n, gate_steps=n, matched_references=0, tracking_steps=0,
                    tracking_active_lane_steps=0, tracking_effective_weight_sum=0., tracking_nonzero_gradient_lanes=0)

    trainer._rollout_objective = rollout
    short = (torch.tensor([2., 4.]), torch.tensor([1, 2]))
    long = (torch.tensor([8., 100., 12.]), torch.tensor([1, 0, 1]))
    packed = pooled_objective(trainer, [short, long], collect_critic_data=False)
    unified = pooled_objective(trainer, [tuple(torch.cat(x) for x in zip(short, long, strict=True))],
                               collect_critic_data=False)
    torch.testing.assert_close(packed['loss'], unified['loss'])
    packed['loss'].backward()
    torch.testing.assert_close(actor.weight.grad, torch.tensor([[.25 * ((2 + 8 + 12) / 3 + 4) / 2]]))


def test_project_actual_adam_step_preserves_common_descent():
    from quadsmash.rl.supermimic_approach_speed_risk import constrain_proposal

    gradients = {'low_speed': torch.tensor([1., 0., 0.]), 'marginal_success': torch.tensor([0., 1., 0.]),
                 'objective': torch.tensor([-1., -1., 4.])}
    step = torch.tensor([1., 1., -2.])
    corrected, diagnostics = constrain_proposal(step, gradients)
    assert diagnostics['applied'] and corrected.norm() <= step.norm() + 1e-6
    assert all(float(g @ corrected) < 0 for g in gradients.values())
    assert diagnostics['normalized_slopes_before']['low_speed'] > 0


def test_projection_allows_tangent_step_for_opposed_constraints():
    from quadsmash.rl.supermimic_approach_speed_risk import constrain_proposal

    g = torch.tensor([1., 0.])
    step, diagnostics = constrain_proposal(torch.tensor([1., -1.]), {'slow': g, 'flight': -g})
    torch.testing.assert_close(step, torch.tensor([0., -1.]), atol=1.e-6, rtol=0.)
    assert diagnostics['requested_margin'] == 0.


def test_projection_ignores_tiny_noise_constraint_gradient():
    from quadsmash.rl.supermimic_approach_speed_risk import constrain_proposal

    obj = torch.tensor([10., 0.])
    tiny_net = torch.tensor([-1.e-3, 1.e-3])  # 1.4e-4 relative to objective -> noise floor
    step = torch.tensor([-1., 0.])
    corrected, diagnostics = constrain_proposal(step, {'objective': obj, 'robot_net': tiny_net})
    assert diagnostics['filtered_noise_keys'] == ['robot_net']
    assert not diagnostics['applied']
    torch.testing.assert_close(corrected, step)


def test_robot_net_clearance_uses_finite_band_and_blocks_new_physical_overlap():
    import itertools
    from quadsmash.physics.court_v4 import load_court
    from quadsmash.rl.supermimic_robot_net_clearance import check_robot_net_clearance, robot_net_clearance

    corners = .01 * torch.tensor(list(itertools.product((-1., 1.), repeat=3)))[None]
    positions = torch.tensor([[[1.48, 0., 1.2]], [[1.5, 0., 1.62]], [[1.5, 4., 1.2]]], requires_grad=True)
    rotation = torch.eye(3).expand(3, 1, 3, 3)
    proxy = (torch.tensor([0]), corners)
    def clearance(p):
        return robot_net_clearance(dict(rotation=rotation, position=p), proxy, load_court())

    old = clearance(positions)
    assert (old > 0).all()
    moved = positions.detach().clone()
    moved[0, 0, 0] = 1.5
    changed = clearance(moved)
    assert changed[0] < 0 and changed[1] > 0
    allowed, receipt = check_robot_net_clearance(old, changed)
    assert not allowed and receipt['new_intersections'] == 1
    old[0].backward()
    assert positions.grad[0, 0, 0] < 0


def test_matched_success_face_speed_loss_is_bounded_and_differentiable(monkeypatch):
    from quadsmash.rl import supermimic_toi_racket_objective as toi

    monkeypatch.setattr(toi, 'racket_features', lambda state, _: state)
    value = torch.tensor([[0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.]], requires_grad=True)
    valid = torch.ones(1, dtype=torch.bool)
    base = dict(matched=valid, toi_racket=value.detach(), intercept=dict(
        time=torch.tensor([.005]), position=torch.zeros(1, 3), incoming_velocity=torch.tensor([[-2., 0., 0.]]),
        valid=valid), ball_only_contact=dict(normal=torch.tensor([[1., 0., 0.]]),
                                            surface_speed=torch.ones(1), valid=~valid))
    def measured(speed):
        return dict(base, measured_face_target=dict(speed=torch.tensor([speed]), normal=torch.tensor([[1., 0., 0.]]),
                  touch_offset=torch.zeros(1, 3), incoming=torch.tensor([[-2., 0., 0.]])))

    def loss(reference):
        return toi.intercept_loss(value, value, torch.zeros(1), torch.tensor([.01]), reference,
                                  valid, torch.ones(1), None, relative_face_speed=True)[0]
    torch.testing.assert_close(loss(measured(3.1)), loss(base))
    assert loss(measured(4.)) > loss(base)
    slope = torch.autograd.grad(loss(measured(4.)), value)[0]
    assert slope[0, 3] < 0
def test_forced_exit_uses_verified_accepted_checkpoint(tmp_path):
    from scripts.run_supermimic_approach_speed_risk import recover_timed_out_update

    preparation = dict(native_ticks=120, actor_sha256='prepared', attempts=21, accepted=20)
    invocation = dict(start=dict(actor_sha256='prepared', attempts=21, accepted=20),
                      actor_updates=2, world_updates=0, critic_updates=0,
                      heads={'recovery': {'accepted': 0}})
    checkpoint = tmp_path / 'approach_A042.pt'
    checkpoint.write_bytes(b'accepted actor')
    receipt = dict(checkpoint=checkpoint.name, retained=42, native_ticks=120, invocation=invocation,
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (tmp_path / 'latest_speed_risk_actor_update.json').write_text(json.dumps(receipt))
    assert recover_timed_out_update(tmp_path, preparation) == (checkpoint, invocation)

    checkpoint.write_bytes(b'partial write')
    with pytest.raises(ValueError, match='checksum'):
        recover_timed_out_update(tmp_path, preparation)


def test_native_gate_uses_clean_repeatability_band():
    from scripts.run_supermimic_approach_speed_risk import native_gate_outcome

    assert native_gate_outcome(-4, 0) == 'rollback'
    assert native_gate_outcome(-1, 1) == 'retain_reference'
    assert native_gate_outcome(1, 1) == 'retain_reference'
    assert native_gate_outcome(2, 1) == 'promote'
