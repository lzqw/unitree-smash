from types import SimpleNamespace

import torch

from quadsmash.rl.frozen_v2_approach_v1 import EarlyArmApproachGate
from quadsmash.rl.supermimic_alternating_schedule_v1 import AlternatingSchedule, make_approach_plan, run_schedule
from quadsmash.rl.supermimic_approach_data_v1 import (
    NativeArmIdentification, comparable_flight_merit, prioritize_flight_gradient,
)
from quadsmash.rl.supermimic_outgoing_flight import calibrated_speed_loss, predict_flight_events
from quadsmash.rl.supermimic_release_calibration import release_error_summary, speed_bias_at_lead
from quadsmash.tasks.stand.approach_ball_terminal import ball_terminal_mask
from scripts.run_supermimic_approach_ball_only import ARCHIVE_INPUTS, milestone_checkpoint, train_command


def test_matched_approach_arms_share_merit_and_budgets_with_distinct_speed_targets():
    registered = [part for name in ARCHIVE_INPUTS for part in ('--' + name, name)]
    control, early = [train_command(registered, 'base.pt', 'phase', motion, stop_at=8, fresh=True,
                                     prioritize_flight=True) for motion in (False, True)]
    speed = '--approach-speed-target'
    assert control[control.index(speed) + 1] == 'historical'
    assert early[early.index(speed) + 1] == 'relative-face'
    assert '--early-arm-seconds' not in control
    gate = early.index('--early-arm-seconds')
    assert early[gate + 1] == '.25'
    assert '--prioritize-approach-flight' in control
    assert control[:control.index(speed)] == early[:gate] + early[gate + 2:early.index(speed)]


def test_final_milestone_finalizes_the_full_schedule_and_preserves_requested_training_seed(tmp_path):
    registered = [part for name in ARCHIVE_INPUTS for part in ('--' + name, name)]
    command = train_command(registered, 'a128.pt', tmp_path, True, stop_at=256, fresh=False,
                            prioritize_flight=True, training_seed=913)
    assert '--approach-stop-at' not in command
    assert '--resume-reset-schedule' not in command
    assert command[command.index('--research-training-seed') + 1] == '913'
    assert milestone_checkpoint(tmp_path, 256) == tmp_path/'latest.pt'
    assert milestone_checkpoint(tmp_path, 128) == tmp_path/'approach_A128.pt'


def test_flight_priority_keeps_unopposed_gradients_and_repairs_uphill_projection():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([-3., 4.])
    flight = torch.tensor([2., 0.])
    assert prioritize_flight_gradient([parameter], [flight]) == 2.5
    torch.testing.assert_close(parameter.grad, torch.tensor([2., 4.]))
    assert prioritize_flight_gradient([parameter], [flight]) == 0.
    torch.testing.assert_close(parameter.grad, torch.tensor([2., 4.]))


def test_flight_line_search_compares_unchanged_baseline_lanes():
    before = torch.tensor([.5, torch.nan, .75])
    improved = torch.tensor([.25, 9., .75])
    worse = torch.tensor([.75, 0., .75])
    missing = torch.tensor([torch.nan, 0., .75])
    assert comparable_flight_merit(before, improved, .5, .4)[0]
    assert not comparable_flight_merit(before, improved, .5, .8)[0]
    assert not comparable_flight_merit(before, worse, .5, .1)[0]
    assert comparable_flight_merit(before, missing, .5, .1)[1]['lost_release']
    assert comparable_flight_merit(before, before, .5, .1)[0]
    assert not comparable_flight_merit(before, before, .5, .8)[0]
    assert comparable_flight_merit(torch.full_like(before, torch.nan), improved, .5, .1)[0]


def test_approach_only_schedule_has_exact_budgets_and_no_recovery_refresh():
    schedule = AlternatingSchedule(make_approach_plan(256, 512))
    calls = []

    def train(event):
        calls.append(event)
        return dict(accepted=event['target'], attempts=event['target'], rejected=0, backtracked=0)

    def refresh(event):
        raise AssertionError(event)

    run_schedule(schedule, train=train, refresh=refresh, finalize=lambda _: {}, checkpoint=lambda _: None)
    assert len(calls) == 32
    assert {(event['head'], event['target']) for event in calls} == {('approach', 8)}
    assert sum(event['proposal_limit'] for event in calls) == 512
    assert schedule.head_totals()['recovery']['accepted'] == 0
    assert schedule.head_totals()['approach']['accepted'] == 256


def test_early_arm_can_apply_a_gradient_while_the_leg_gate_is_zero():
    raw = torch.zeros((2, 794))
    raw[:, 61] = 1.
    raw[:, 64] = -5.
    raw[:, 67] = 1.
    raw[1, 773] = 1.
    residual = torch.ones((2, 18), requires_grad=True)
    gate = EarlyArmApproachGate()(raw, torch.zeros((2, 1)))
    (residual * gate).sum().backward()
    assert (residual.grad[0, :12] == 0).all()
    assert (residual.grad[0, 12:] > 0).all()
    assert (residual.grad[1] == 0).all()


def test_arm_identification_only_changes_selected_training_lanes_for_four_ticks():
    pulse = NativeArmIdentification(3, 'cpu', excluded=[1], stride=1)
    observation = torch.zeros((3, 794))
    observation[:, 61] = 1.
    observation[:, 64] = -5.
    observation[:, 67] = 1.
    episode = torch.tensor([0, 0, 4])
    no_contact = torch.zeros(3, dtype=torch.bool)
    action = torch.zeros((3, 18))
    action[1, 12:] = 1.2
    expected = action.clone()
    expected[0, 12] = .01
    expected[2, 12] = -.01

    for _ in range(4):
        torch.testing.assert_close(pulse.apply(action, observation, episode, no_contact), expected)
    torch.testing.assert_close(pulse.apply(action, observation, episode, no_contact), action)
    assert pulse.excited.tolist() == [True, False, True]
    episode += 1
    torch.testing.assert_close(pulse.apply(action, observation, episode, ~no_contact), action)
    assert not pulse.excited.any()


def test_ball_phase_retains_recent_native_starts_for_delayed_outcome_labels():
    from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank

    bank = ActorStartBank(capacity=3, ball_phase=True)
    active = torch.ones(4, dtype=torch.bool)
    for tick in range(16):
        start = dict(stream_id=2, before=dict(episode_index=torch.zeros(4, dtype=torch.long),
                     elapsed_tick=torch.full((4,), tick), physical=dict(elapsed_s=torch.full((4,), tick / 100))))
        bank.append(start, dict(approach_gate=torch.full((4,), .1), had_contact=~active,
                               approach_start_eligible=active), stream=2, policy_version=0, native_tick=tick)

    assert [entry['native_tick'] for entry in bank.entries] == [4, 8, 12]
    terminal = dict(scoring_schema='approach_ball_only_v1', ball_side_complete_except_recovery=False,
                    env_id=1, episode_index=0, qualified_airborne_string_contact=True,
                    approach_ball_terminal=dict(numerical_abort=False), outgoing_speed_mps=4.,
                    rejection_reasons=['speed_threshold'], contact_time_s=.25)
    bank.label_terminal(terminal, 2)
    _, sampled = bank.sample_ball_phase(lambda _: active, 0)
    assert sampled['actual_by_outcome']['low_speed'] == 1
    assert sampled['outcome_shortfall']['low_speed'] == 5
    assert sampled['effective_lanes'] == 4

    restored = ActorStartBank(capacity=3, ball_phase=True)
    restored.load_state_dict(bank.state_dict())
    assert restored.sampling_report()['actual_by_outcome']['low_speed'] == 1


def test_native_phase_timing_preserves_substeps_and_restores_simulator_method():
    from quadsmash.rl.supermimic_native_phase_timing import NativePhaseTiming

    calls = []

    def sim_step(*, render):
        calls.append(render)

    sim = SimpleNamespace(step=sim_step)
    rally = SimpleNamespace(had_contact=torch.tensor([False]), released=torch.tensor([False]),
                            ground_seen=torch.tensor([False]))
    timing = NativePhaseTiming(SimpleNamespace(sim=sim, rally=rally), samples_per_phase=1, stride=1)
    observation = torch.zeros((1, 794))

    def advance(action):
        sim.step(render=False)
        sim.step(render=True)
        return action

    action = torch.zeros(1, 18)
    assert timing.step(advance, action, observation, 0) is action
    observation[:, 67] = 1.
    assert timing.step(advance, action, observation, 1) is action
    rally.had_contact[:] = rally.released[:] = True
    assert timing.step(advance, action, observation, 2) is action
    assert calls == [False, True] * 3 and sim.step is sim_step
    assert all(len(rows) == 1 and rows[0]['physics_substeps'] == 2
               for rows in timing.report()['samples'].values())


def test_ball_flight_finishes_after_robot_fall_and_retains_faulted_flight():
    yes = torch.tensor([True])
    no = ~yes
    base = dict(robot_terminal=yes, had_contact=yes, released=yes, ground_seen=no,
                physical_ground_seen=no, faulted=no, followthrough_pending=no,
                time_since_release=torch.tensor([.5]), followthrough_seconds=.04, timeout=no)
    assert not ball_terminal_mask(**base)
    assert ball_terminal_mask(**dict(base, had_contact=no))
    assert ball_terminal_mask(**dict(base, ground_seen=yes))
    assert not ball_terminal_mask(**dict(base, faulted=yes))
    assert ball_terminal_mask(**dict(base, faulted=yes, physical_ground_seen=yes))


def test_unlanded_flight_does_not_fabricate_a_landing_and_keeps_release_gradient():
    prior = SimpleNamespace(preview=SimpleNamespace(radius=.02), gravity=torch.tensor([0., 0., -9.81]),
                            wind=torch.zeros(3), drag=0.)
    court = dict(net_plane_x_m=1., net_center_top_m=1., net_side_top_m=1., post_y_center_abs_m=3.)
    velocity = torch.tensor([[4., 0., 20.], [4., 0., 0.]], requires_grad=True)
    position = torch.tensor([[0., 0., 1.], [0., 0., 1.]])
    events = predict_flight_events(position, velocity, prior, court, duration=2.5)
    assert events['landed'].tolist() == [False, True]
    assert events['timeout'].tolist() == [True, False]
    assert events['landing_position'][0].isnan().all()
    assert events['landing_time'][0].isnan()
    events['landing_position'][1, 0].backward()
    assert velocity.grad[1, 0] > 0


def test_blocking_net_contact_reflects_without_a_false_opponent_landing():
    prior = SimpleNamespace(preview=SimpleNamespace(radius=.02), gravity=torch.tensor([0., 0., -9.81]),
                            wind=torch.zeros(3), drag=0., restitution=.5)
    court = dict(net_plane_x_m=1., net_center_top_m=1.5, net_side_top_m=1.5,
                 post_y_center_abs_m=3., net_width_m=6., net_depth_m=.76)
    flight = predict_flight_events(torch.tensor([[.8, 0., 1.2]]), torch.tensor([[5., 0., 0.]]), prior, court)
    assert flight['net_contact'].item()
    assert not flight['crossed_net'].item()
    assert flight['net_clearance'].item() < 0.
    assert flight['landing_position'][0, 0] < court['net_plane_x_m']


def test_speed_correction_restores_low_speed_gradient_without_training_bias():
    velocity = torch.tensor([[5.1, 0., 0.]], requires_grad=True)
    bias = torch.tensor(.8, requires_grad=True)
    calibrated_speed_loss(velocity, bias).sum().backward()
    assert velocity.grad[0, 0] < -.5
    assert bias.grad is None


def test_release_bias_is_signed_and_uses_only_supported_leads():
    status = release_error_summary(torch.tensor([4.8, 5.2, 6.]), torch.tensor([4.2, 4.4, 5.8]))
    assert status['matched'] == 3 and status['false_pass'] == 2
    assert status['corrected_false_pass'] < status['false_pass']
    calib = dict(by_lead=[dict(lead_ticks=4, bias_mps=-.2), dict(lead_ticks=8, bias_mps=None),
                         dict(lead_ticks=16, bias_mps=.9)])
    torch.testing.assert_close(speed_bias_at_lead(torch.tensor([0, 8, 14, 30]), calib),
                               torch.tensor([-.2, -.2, .9, .9]))


def test_approach_evaluation_boundary_keeps_the_same_runnable_cursor():
    plan = make_approach_plan(256, 512)
    schedule = AlternatingSchedule(plan)

    def accepted(event):
        return dict(accepted=event['target'], attempts=event['target'], rejected=0, backtracked=0)

    complete = run_schedule(
        schedule, train=accepted, refresh=lambda _: None, finalize=lambda _: {},
        checkpoint=lambda _: None, stop_after=lambda _, state: state.head_totals()['approach']['accepted'] == 64)
    assert not complete
    assert schedule.current['block_key'] == 'approach_only_008'
    restored = AlternatingSchedule(plan)
    restored.load_state_dict(schedule.state_dict())
    assert restored.current == schedule.current
