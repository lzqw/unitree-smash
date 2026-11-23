import pytest
import torch

from quadsmash.rl.supermimic_approach_data_v1 import approach_guard_merit, approach_step_sizes
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1


def test_guard_checks_aggregate_and_preserves_large_string_margins():
    a = dict(flight_cost_by_lane=torch.tensor([.1, .1, .1, .1]),
             string_margin_by_lane=torch.tensor([.02, .02, .02, .02]), loss=.5)
    b = dict(a, flight_cost_by_lane=torch.tensor([.101, .08, .09, .09]),
             string_margin_by_lane=torch.tensor([.019, .019, .021, .022]), loss=.4)
    assert approach_guard_merit(a, b)[0]
    lost = dict(b, flight_cost_by_lane=torch.tensor([torch.nan, .08, .09, .09]))
    assert not approach_guard_merit(a, lost)[0]
    margin = dict(b, string_margin_by_lane=torch.tensor([.011, .012, .02, .02]))
    assert not approach_guard_merit(a, margin)[0]
    assert approach_step_sizes(.25) == (.25, .5, .125, 1., .0625)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA discrete adjoint')
@pytest.mark.parametrize('trajectory', [False, True])
def test_cuda_flight_values_and_adjoint_match_eager(trajectory):
    pytest.importorskip('triton')
    from quadsmash.rl.supermimic_differentiable_flight_cuda import DifferentiableCudaImpulsePrior

    eager, fused = BallImpulsePriorV1().cuda(), DifferentiableCudaImpulsePrior().cuda()
    generator = torch.Generator(device='cuda').manual_seed(72)
    n = 35
    position = torch.randn((n, 6), generator=generator, device='cuda')[:, :3]
    velocity = torch.randn((n, 3), generator=generator, device='cuda') * 7
    velocity[0] = eager.wind
    duration = torch.linspace(0, eager.dt, n, device='cuda')
    duration[1] = eager.sub_dt
    shape = (n, eager.steps + 1, 3) if trajectory else (n, 3)
    weights = tuple(torch.randn(shape, generator=generator, device='cuda') for _ in range(2))

    def compare(model):
        p, v, t = [value.detach().requires_grad_(True) for value in (position, velocity, duration)]
        out = model.advance_trajectory(p, v) if trajectory else model.advance(p, v, t)
        grad = torch.autograd.grad(sum((x * w).sum() for x, w in zip(out, weights, strict=True)),
                                   (p, v) if trajectory else (p, v, t))
        return (*out, *grad)

    for a, b in zip(compare(fused), compare(eager), strict=True):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=4e-4)


def test_delayed_native_label_survives_fifo_and_reloads_disjoint_batches():
    from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank

    bank = ActorStartBank(capacity=2, ball_phase=True, persistent_events=True)
    n = 56
    gate = torch.ones(n)
    capture = {'approach_gate': gate, 'had_contact': torch.zeros(n)}
    for tick in (100, 104, 108, 112, 116, 120, 124):
        snapshot = dict(
            stream_id=1, before=dict(episode_index=torch.zeros(n, dtype=torch.long),
                                      elapsed_tick=torch.full((n,), tick, dtype=torch.long),
                                      physical=dict(elapsed_s=torch.full((n,), tick * .01))))
        bank.append(snapshot, capture, stream=1, policy_version=32, native_tick=tick)
    assert min(row['native_tick'] for row in bank.entries) > 110
    for lane in range(n):
        bank.label_terminal(dict(
            env_id=lane, episode_index=0, case_id=f'train-{lane}', scoring_schema='approach_ball_only_v1',
            outgoing_speed_mps=4.1 if lane % 2 else 5.5, qualified_airborne_string_contact=True,
            contact_time_s=1.18, ball_side_complete_except_recovery=lane % 2 == 0,
            rejection_reasons=['speed_threshold'] if lane % 2 else [],
            approach_ball_terminal={'numerical_abort': False}), 1)
    assert len(bank.resolved) == n and not bank.pending
    assert min(row['native_tick'] for row in bank.resolved[(1, 0, 0)][0]) == 100
    restored = ActorStartBank(capacity=2, ball_phase=True, persistent_events=True)
    restored.load_state_dict(bank.state_dict())
    _, _, report = restored.sample_proposal_and_guard(lambda start: gate.bool(), 1, 64, 32)
    assert report['effective_lanes'] == 32 and report['guard']['effective_lanes'] == 16
    def ids(draw):
        return {(stream, lane, episode) for record in draw['records']
                for lane, episode, _, _, _, stream in record['lane_episode_tick']}
    assert len(ids(report)) == 32 and len(ids(report['guard'])) == 16
    assert ids(report).isdisjoint(ids(report['guard']))
