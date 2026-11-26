import pytest
import torch

from quadsmash.rl.supermimic_hybrid_native import (
    load_training_data, native_layout, pack_tapes, perturbation_envelope)


def test_native_panel_is_unperturbed_and_teacher_lanes_exclude_holdout():
    cases = [{'case_id': i} for i in range(128)]
    indices = list(range(1, 32, 2))
    spec = dict(case_indices=indices, offsets=[[0.] * 6, [.01] * 6, [-.01] * 6, [.02] * 6])
    ordered = native_layout(cases, spec, {33})
    assert ordered[:64] == cases[::2]
    for offset in range(64, 128, 16):
        assert ordered[offset:offset + 16] == [cases[i] for i in indices]
    with pytest.raises(ValueError, match='distinct training cases'):
        native_layout(cases, spec, {31})
    assert perturbation_envelope(226) == 0.
    assert perturbation_envelope(267) == 1.
    assert perturbation_envelope(308) == 0.


def test_teacher_uses_own_observation_and_requires_real_execution_change(tmp_path):
    indices = list(range(1, 32, 2))

    def result(case, success):
        return dict(case_id=case, success=success, string=success, robot_net=False,
                    ball_net=False, low_speed=False, landed_target=success,
                    margin_score=.3 if success else -.2)

    outcomes = [dict(rows=[result(case, i == 1 or branch == 1 and i in (0, 2))
                           for i, case in enumerate(indices)]) for branch in range(4)]
    targets = torch.zeros(64, 18)
    targets[16, 13] = .01  # case 0 changed an executed target, case 2 did not
    frame = dict(episode=torch.zeros(64, dtype=torch.long), had_contact=torch.zeros(64, dtype=torch.bool),
                 present=torch.ones(64, dtype=torch.bool), gate=torch.ones(64),
                 observation=torch.arange(64.)[:, None], request=targets.clone(), targets=targets)
    dataset = pack_tapes([frame], outcomes, indices, torch.zeros(64, 6), 'actor-sha')
    assert dataset['examples'] == {'teacher': [16], 'preserve': [1]}
    path = tmp_path / 'tape.pt'
    torch.save(dataset, path)
    supervision = load_training_data(path, {33})['supervision']
    assert supervision['teacher']['observation'].item() == 16
    assert supervision['preserve']['observation'].item() == 1
    assert supervision['teacher']['targets'][0, 13] == .01


def test_quick_actuator_contract_compares_effective_lane_profiles():
    from run_supermimic_approach_hybrid_boundary import quick_actuator_contract

    normal, alternate = {'kp': [80.]}, {'kp': [65.]}
    baseline = dict(schema='native-joint-drive-v1', profiles=[normal], lane_profile=[0] * 64)
    mixed = dict(schema=baseline['schema'], profiles=[alternate, normal],
                 lane_profile=[1] * 64 + [0] * 64)
    assert quick_actuator_contract(baseline) == quick_actuator_contract(mixed)
    mixed['lane_profile'][17] = 0
    assert quick_actuator_contract(baseline) != quick_actuator_contract(mixed)


def test_native_actor_loss_anchors_legs_on_preserve_and_uses_sharp_weights():
    from quadsmash.rl.supermimic_hybrid_native import native_actor_loss

    class DummyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(18))

        def forward_with_recovery_gate(self, obs, no_rec, sample=False):
            out = self.w[None].expand(len(obs), 18)
            return out, None, None

    actor = DummyActor()
    req = torch.zeros(4, 18)
    req[:, 0] = 0.05  # leg offset on preserve
    req[:, 12] = 0.05  # arm offset
    samples = {
        'teacher': {'observation': torch.zeros(4, 2), 'request': req.clone()},
        'preserve': {'observation': torch.zeros(4, 2), 'request': req.clone()},
    }
    loss, stats = native_actor_loss(actor, samples, attempt=1)
    assert 'native_preserve_leg_loss' in stats and stats['native_preserve_leg_loss'] > 0.
    loss.backward()
    assert actor.w.grad[0].abs() > 0. and actor.w.grad[12].abs() > 0.
