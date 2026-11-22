from types import SimpleNamespace

import torch

from quadsmash.rl.supermimic_alternating_schedule_v1 import make_plan
from quadsmash.rl.supermimic_outgoing_flight import outgoing_cost
from quadsmash.rl.supermimic_training_stage_v1 import resolve_training_stage


def test_256_retained_and_two_combined_refreshes():
    events=make_plan(128,128,256,256,refresh_every_pairs=8)
    blocks=[e for e in events if e['kind']=='train']
    assert [e['head'] for e in blocks]==['recovery','approach']*16
    assert sum(e['target'] for e in blocks)==256
    assert sum(e['proposal_limit'] for e in blocks)==512
    assert [e['pair'] for e in events if e['kind']=='refresh']==[7,15]
    stage=resolve_training_stage('research_contact')
    assert stage.contact_bptt_mode=='enabled' and not stage.promotable


def test_outgoing_cost_weak_return_has_forward_descent_direction():
    prior=SimpleNamespace(gravity=torch.tensor([0.,0.,-9.81]),wind=torch.zeros(3),drag=.02)
    court=dict(net_plane_x_m=1.5,target_x_limits_m=[2.,4.],target_y_half_width_m=1.,
               net_side_top_m=1.55,net_center_top_m=1.524,post_y_center_abs_m=3.075)
    ball=torch.tensor([[0.,0.,2.,1.,0.,1.]],requires_grad=True)
    cost=outgoing_cost(ball,prior,court)
    g,=torch.autograd.grad(cost.sum(),ball)
    plus=ball.detach().clone()
    plus[:,3]+=.001
    torch.testing.assert_close(g[:,3],(outgoing_cost(plus,prior,court)-cost.detach())/.001,atol=.002,rtol=.003)
    assert g[0,3]<0


def test_only_first_string_flight_can_earn_outgoing_credit():
    from quadsmash.rl.supermimic_outgoing_flight import first_string_flight_loss
    prior = SimpleNamespace(gravity=torch.tensor([0., 0., -9.81]), wind=torch.zeros(3),
                            drag=.02, preview=SimpleNamespace(radius=.02))
    court = dict(net_plane_x_m=1.5, target_x_limits_m=[2., 4.], target_y_half_width_m=1.,
                 net_side_top_m=1.55, net_center_top_m=1.524, post_y_center_abs_m=3.075)
    ball = torch.tensor([[0., 0., 2., 3.5, 0., 1.]] * 4, requires_grad=True)
    selected = torch.tensor([True, True, True, False])
    first_string = torch.tensor([True, False, False, True])
    loss = first_string_flight_loss(ball, selected, first_string, prior, court)
    legal_cost = -torch.expm1(-outgoing_cost(ball[:1], prior, court, scoring_margins=True))
    torch.testing.assert_close(loss, (legal_cost.sum() + 2.) / 3.)
    grad, = torch.autograd.grad(loss, ball)
    assert grad[0].norm() > 0 and grad[1:].count_nonzero() == 0
    no_strings = first_string_flight_loss(ball, selected, torch.zeros_like(selected), prior, court)
    assert no_strings.item() == 1. and torch.autograd.grad(no_strings, ball)[0].count_nonzero() == 0


def test_downward_release_uses_native_threshold_and_only_legal_string_lanes():
    from quadsmash.rl.supermimic_outgoing_flight import first_string_flight_loss
    prior = SimpleNamespace(gravity=torch.tensor([0., 0., -9.81]), wind=torch.zeros(3),
                            drag=.02, preview=SimpleNamespace(radius=.025))
    court = dict(net_plane_x_m=1.5, target_x_limits_m=[2., 4.], target_y_half_width_m=1.,
                 net_side_top_m=1.55, net_center_top_m=1.524, post_y_center_abs_m=3.075)
    ball = torch.tensor([[1., 0., 2., 6., 0., vz] for vz in (-.75, -1., -1.25, -.75)],
                        requires_grad=True)
    old = outgoing_cost(ball, prior, court, scoring_margins=True)
    new = outgoing_cost(ball, prior, court, scoring_margins=True, release_velocity=ball[:, 3:])
    torch.testing.assert_close(new - old, torch.tensor([.05, 0., 0., .05]))
    g, = torch.autograd.grad((new - old).sum(), ball)
    expected = torch.zeros_like(g)
    expected[[0, 3], 5] = .2
    torch.testing.assert_close(g, expected)
    # A shallow release can become steep enough by the next control boundary.
    endpoint = ball.detach().clone()
    endpoint[:, 5] = -1.25
    release = endpoint[:, 3:].clone().requires_grad_()
    with torch.no_grad():
        release[0, 2] = -.75
        release[1, 0] = 2.
    matched = outgoing_cost(endpoint, prior, court, scoring_margins=True, release_velocity=release)
    direct = (5.25 - release.norm(dim=-1)).relu()/5 + (release[:, 2]+1.).relu()/5
    torch.testing.assert_close(matched-outgoing_cost(endpoint, prior, court, scoring_margins=True), direct)
    assert torch.autograd.grad(matched.sum(), release)[0][0, 2] > 0
    legal = torch.tensor([True, True, True, False])
    cost = first_string_flight_loss(ball, torch.ones_like(legal), legal, prior, court,
                                    release_velocity=ball[:, 3:])
    torch.testing.assert_close(cost, ((-torch.expm1(-new[legal])).sum()+1.)/4)
    assert torch.autograd.grad(cost, ball)[0][3].count_nonzero() == 0


def test_flight_cost_keeps_airborne_lanes_after_other_lanes_land():
    prior = SimpleNamespace(gravity=torch.tensor([0., 0., -9.81]), wind=torch.zeros(3), drag=.02)
    court = dict(net_plane_x_m=1.5, target_x_limits_m=[2., 4.], target_y_half_width_m=1.,
                 net_side_top_m=1.55, net_center_top_m=1.524, post_y_center_abs_m=3.075)
    ball = torch.tensor([[0., 0., .05, 1., 0., -2.], [0., 0., 2., 1., 0., .5]], requires_grad=True)
    full = outgoing_cost(ball, prior, court)
    short = outgoing_cost(ball, prior, court, steps=16)
    assert full[1] != short[1]
    separate = torch.cat([outgoing_cost(ball[:1], prior, court, steps=16),
                          outgoing_cost(ball[1:], prior, court)])
    torch.testing.assert_close(full, separate)
    torch.testing.assert_close(torch.autograd.grad(full.sum(), ball)[0], torch.autograd.grad(separate.sum(), ball)[0])


def test_research_replicate_forks_sampler_streams_and_resume_keeps_cursors():
    import pytest
    from scripts.train_supermimic_physical_block import research_learner_rng

    replay = SimpleNamespace(generator=torch.Generator(), actor_start_bank=SimpleNamespace(generator=torch.Generator()))
    flight = SimpleNamespace(generator=torch.Generator())

    def sample():
        return [torch.randperm(32, generator=g).tolist() for g in (None, replay.generator, flight.generator, replay.actor_start_bank.generator)]

    def start(seed):
        assert research_learner_rng(seed, {}, fresh=True, replay=replay, natural_flight=flight) == seed
        return sample()

    with torch.random.fork_rng(devices=[]):
        first = start(321)
        second = sample()
        assert all(a != b for a, b in zip(first, start(322)))
        assert first == start(321)
        config = {'research_training_seed': 321}
        assert research_learner_rng(321, config, fresh=False, replay=replay, natural_flight=flight) == 321
        assert research_learner_rng(None, config, fresh=False, replay=replay, natural_flight=flight) == 321
        assert sample() == second
        with pytest.raises(ValueError, match='ongoing schedule'):
            research_learner_rng(322, config, fresh=False, replay=replay, natural_flight=flight)


def test_extension_launches_from_final_refresh_and_preserves_original_axes(tmp_path):
    import json
    import pytest
    from scripts.extend_supermimic_research_scale import extension_registration
    from scripts.report_supermimic_research_curve import arm_origin
    from scripts.run_supermimic_research_scale import digest

    parent = tmp_path/'main'
    train = parent/'train'
    archive = train/'native_transitions'
    archive.mkdir(parents=True)
    manifest = {'chunks': [{'sha256': 'chunk'}]}
    (archive/'manifest.json').write_text(json.dumps(manifest))
    contact = dict(**manifest, durable_env_steps=91008, total={'frame':3,'qualified_contact':6,'qualified_or_frame':9})
    (parent/'native_contact_counts.json').write_text(json.dumps(contact))
    initial = tmp_path/'original.pt'
    torch.save(dict(global_counters={'accepted':17},native_ticks=4990), initial)
    reg = dict(arm='G1',expected_retained_updates=256, initial={'checkpoint_sha256':digest(initial)},
               command=['train','--resume-training',str(initial),'--resume-reset-schedule','--cycles','1',
                        '--output',str(train),'--tensorboard-dir','old',
                        '--landing-and-first-contact-objectives','--wm-stage-quota','uniform=108,post_landing_support_loss=20'])
    (parent/'registration.json').write_text(json.dumps(reg))
    (parent/'exit.json').write_text(json.dumps({'returncode':0,'wall_seconds':1000}))
    completed=dict(actor_updates=256,recovery_accepted=128,approach_accepted=128,global_accepted=273,
                   world_updates=490,critic_updates=560,head_counters={'recovery':{'attempts':130},'approach':{'attempts':138}})
    (train/'training_completed.json').write_text(json.dumps(completed))
    ck = dict(config={'research_origin':17}, global_counters={'accepted':273},world_updates=490,critic_updates=560,
              alternating_schedule=dict(running=False,cursor=1,plan=[{'kind':'finalize'}],results=[{'event':{'kind':'finalize'}}]))
    torch.save(ck, train/'latest.pt')
    extended = extension_registration(parent, tmp_path/'next', tmp_path/'tb', 1024)
    assert arm_origin(extended) == (17, 4990)
    assert extended['previous_heads'] == {'recovery':128,'approach':128}
    assert extended['previous_proposals'] == 268
    cmd = extended['command']
    assert cmd[cmd.index('--resume-training')+1] == str(train/'latest.pt')
    assert cmd[cmd.index('--cycles')+1] == '4' and '--resume-reset-schedule' not in cmd
    assert cmd.count('--restart-uphill-approach-momentum') == 1
    quota = dict(item.split('=') for item in cmd[cmd.index('--wm-stage-quota')+1].split(','))
    assert quota == {'uniform':'108','post_landing_support_loss':'16','post_landing_unload':'4'}
    assert cmd.count('--recovery-rollout-landing') == cmd.count('--score-downward-release') == 1
    ck['world_updates'] -= 4
    torch.save(ck, train/'latest.pt')
    with pytest.raises(ValueError, match='post-refresh'):
        extension_registration(parent, tmp_path/'next', tmp_path/'tb', 1024)


def test_approach_draws_span_short_blocks_rejections_and_resume():
    from collections import Counter
    from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank
    from quadsmash.rl.supermimic_sequential_progress_v1 import SequentialProgress

    schedule = SequentialProgress()
    bank = ActorStartBank()
    requested = []
    for block, count in enumerate([8, 9, 3]):
        schedule.block(block, 'recovery', 8, 16)['attempts'] = 10
        for i in range(1, count+1):
            progress = schedule.block(block, 'approach', 8, 16)
            progress['attempts'] = i
            progress['accepted'] = min(i, 8)
            if block == 1 and i == 4:
                schedule = SequentialProgress.for_run(schedule.state_dict(), fresh=False)
            draw = schedule.head_attempts('approach')
            assert draw == len(requested)+1
            requested.append(bank.sample(draw)[1]['requested'])
    assert Counter(requested) == {'gate_onset':5, 'pre_contact':7, 'failure_pre':5, 'success_near':3}


def test_matched_control_copies_revised_flags_seed_and_handoff_identity(tmp_path):
    from scripts.continue_supermimic_research_comparison import control_registration
    from scripts.run_supermimic_research_scale import digest
    import pytest
    handoff,contact=tmp_path/'handoff.pt',tmp_path/'contact.pt'
    handoff.write_bytes(b'scored');contact.write_bytes(b'calibrated')
    cmd=['train.py','--research-contact-gradient','G1','--resume-training',str(handoff),
         '--action-ball-checkpoint',str(contact),'--output','original/train','--tensorboard-dir','original/tb',
         '--research-training-seed','2718','--interleaved-approach-starts','--restart-uphill-approach-momentum']
    reg=dict(arm='G1',initial=dict(checkpoint_sha256='experiment origin',contact_checkpoint_sha256=digest(contact)),
             revision=dict(checkpoint_sha256=digest(handoff)),command=cmd,research_training_seed=2718)
    control=control_registration(reg,tmp_path/'control',tmp_path/'tb')
    for key in ('initial','revision','research_training_seed'):
        assert control[key]==reg[key]
    assert control['command'][1:3]==['--research-contact-gradient','G0'] and cmd[2]=='G1'
    assert control['command'][-4:]==cmd[-4:]
    handoff.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):
        control_registration(reg,tmp_path/'control',tmp_path/'tb')
