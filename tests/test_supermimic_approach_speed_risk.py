import json
from pathlib import Path

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
