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


def test_research_replicate_forks_sampler_streams_and_resume_keeps_cursors():
    import pytest
    from scripts.train_supermimic_physical_block import research_learner_rng

    replay = SimpleNamespace(generator=torch.Generator())
    flight = SimpleNamespace(generator=torch.Generator())

    def sample():
        return [torch.randperm(32, generator=g).tolist() for g in (None, replay.generator, flight.generator)]

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
    quota = dict(item.split('=') for item in cmd[cmd.index('--wm-stage-quota')+1].split(','))
    assert quota == {'uniform':'108','post_landing_support_loss':'16','post_landing_unload':'4'}
    assert cmd.count('--recovery-rollout-landing') == 1
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
