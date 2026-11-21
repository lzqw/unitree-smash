"""CPU orchestration contracts only: no native episode or real checkpoint updates."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from quadsmash.rl.supermimic_alternating_schedule_v1 import (
    AlternatingSchedule,make_plan,refresh_world_budgets,run_schedule,
)
from quadsmash.rl.supermimic_sequential_progress_v1 import SequentialProgress


def accepted(event, extra=0):
    return dict(accepted=event['target'],attempts=event['target']+extra,rejected=extra,backtracked=0)


def refreshed(event):
    return dict(complete_episodes=128,actor_unchanged=True,consumer_heads=['recovery','approach'])


def test_r64_a64_means_eight_combined_refreshes_and_one_finalization():
    p=make_plan(64,64,128,128)
    assert len(p)==25
    assert [(e['kind'],e.get('head')) for e in p[:3]]==[('train','recovery'),('train','approach'),('refresh',None)]
    assert sum(e['kind']=='refresh' for e in p)==8
    for h in ('recovery','approach'):
        assert sum(e['target'] for e in p if e.get('head')==h)==64
        assert sum(e['proposal_limit'] for e in p if e.get('head')==h)==128
    assert p[-1]['kind']=='finalize'
    assert sum(e['kind']=='refresh' for e in make_plan(64,64,128,128,refresh_mode='per-head'))==16


def test_remainder_and_proposal_surplus_not_rounded_into_extra_updates():
    p=make_plan(10,10,13,15,cycles=2)
    for c in (0,1):
        for head,limit in [('recovery',13),('approach',15)]:
            blocks=[e for e in p if e.get('head')==head and e['cycle']==c]
            assert [e['target'] for e in blocks]==[8,2]
            assert sum(e['proposal_limit'] for e in blocks)==limit
    with pytest.raises(ValueError):make_plan(64,64,63,128)
    with pytest.raises(ValueError):make_plan(64,8,128,16)
    with pytest.raises(ValueError):make_plan(8,8,16,16,block_size=0)
    assert refresh_world_budgets(32)==dict(recovery=16,approach=16)
    assert sum(refresh_world_budgets(33).values())==33
    with pytest.raises(ValueError):refresh_world_budgets(1)


def test_driver_uses_current_combination_and_refresh_precedes_next_recovery():
    s=AlternatingSchedule(make_plan(16,16,32,32));version={'recovery':0,'approach':0};seen=[];saves=[]
    def train(e):
        seen.append((e['head'],dict(version)))
        version[e['head']]+=e['target']
        return accepted(e,1)
    def refresh(e):
        seen.append(('refresh',dict(version)))
        return dict(refreshed(e),policy_id=str(version))
    run_schedule(s,train=train,refresh=refresh,finalize=lambda e:{},checkpoint=saves.append)
    assert seen[2]==('refresh',{'recovery':8,'approach':8})
    assert seen[3]==('recovery',{'recovery':8,'approach':8})
    assert seen[-1]==('refresh',{'recovery':16,'approach':16})
    assert s.complete and s.head_totals()=={h:dict(attempts=18,accepted=16,rejected=2,backtracked=0) for h in version}
    assert sum(n.startswith('pre_refresh_') for n in saves)==2


def test_incomplete_head_does_not_run_next_head_or_refresh():
    s=AlternatingSchedule(make_plan(8,8,16,16));calls=[]
    def failed(e):return dict(accepted=0,attempts=16,rejected=16,backtracked=0)
    with pytest.raises(ValueError,match='incomplete'):
        run_schedule(s,train=failed,refresh=lambda e:calls.append(e),finalize=lambda e:calls.append(e),checkpoint=lambda n:None)
    assert not calls and not s.complete and s.cursor==0


def test_resume_at_second_head_does_not_repeat_recovery_or_change_plan():
    plan=make_plan(16,16,32,32);s=AlternatingSchedule(plan)
    s.begin();s.finish(accepted(s.current,1));s.begin()
    saved=s.state_dict();t=AlternatingSchedule(plan);t.load_state_dict(saved)
    assert t.current['head']=='approach' and t.running
    calls=[]
    run_schedule(t,train=lambda e:(calls.append(e['head']) or accepted(e)),refresh=refreshed,finalize=lambda e:{},checkpoint=lambda n:None)
    assert calls==['approach','recovery','approach']
    assert t.head_totals()['recovery']['attempts']==17
    with pytest.raises(ValueError,match='configuration'):
        AlternatingSchedule(make_plan(16,16,64,32)).load_state_dict(saved)


def test_partial_refresh_is_not_resumed_as_a_completed_or_repeated_fit():
    s=AlternatingSchedule(make_plan(8,8,16,16));durable={}
    def checkpoint(name):durable[name]=s.state_dict()
    def fail(e):raise RuntimeError('synthetic WM fit interruption')
    with pytest.raises(RuntimeError):
        run_schedule(s,train=accepted,refresh=fail,finalize=lambda e:{},checkpoint=checkpoint)
    assert s.running and s.current['kind']=='refresh'
    with pytest.raises(ValueError,match='pre_refresh'):
        AlternatingSchedule(s.plan).load_state_dict(s.state_dict())
    safe=AlternatingSchedule(s.plan);safe.load_state_dict(durable['latest.pt'])
    assert safe.current['kind']=='refresh' and not safe.running
    run_schedule(safe,train=lambda e:pytest.fail('repeated training'),refresh=refreshed,finalize=lambda e:{},checkpoint=lambda n:None)
    assert safe.complete


def test_resume_after_refresh_does_not_collect_or_fit_it_again():
    s=AlternatingSchedule(make_plan(8,8,16,16));durable={}
    def checkpoint(n):durable[n]=s.state_dict()
    def fail_finalize(e):raise RuntimeError('report interruption')
    with pytest.raises(RuntimeError):run_schedule(s,train=accepted,refresh=refreshed,finalize=fail_finalize,checkpoint=checkpoint)
    t=AlternatingSchedule(s.plan);t.load_state_dict(durable['latest.pt'])
    assert t.current['kind']=='finalize'
    run_schedule(t,train=lambda e:pytest.fail('train repeated'),refresh=lambda e:pytest.fail('refresh repeated'),finalize=lambda e:{},checkpoint=lambda n:None)
    assert t.complete
    bad=t.state_dict();bad['results'][2]['result']['complete_episodes']=127
    with pytest.raises(ValueError,match='Incomplete'):AlternatingSchedule(s.plan).load_state_dict(bad)


def test_actual_entry_callbacks_disable_internal_refresh_and_use_one_shared_collection():
    # Execute the native entry's actual orchestration callbacks without importing Isaac.
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/train_supermimic_physical_block.py').read_text();tree=ast.parse(source)
    names={'train_short','refresh_pair','finalize_cycle'}
    funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    schedule=SequentialProgress();alt=AlternatingSchedule(make_plan(16,16,32,32));calls=[];files=[]
    def block(h,names,target,limit,key,*,managed_refresh):
        assert managed_refresh
        assert names==[h]
        calls.append(('train',h,key))
        schedule.block(key,h,target,limit).update(accepted=target,attempts=target,rejected=0,backtracked=0,status='done')
    def refresh(head,*,event_id,managed_refresh):
        assert head=='combined' and managed_refresh
        calls.append(('refresh',event_id))
        return dict(refreshed({}),world_updates=32,critic_updates=20)
    ns=dict(run_block=block,schedule=schedule,alternating=alt,RECOVERY_NAMES=['recovery'],APPROACH_NAMES=['approach'],
        refresh_frozen_policy=refresh,trainer=SimpleNamespace(world_updates=64,critic_updates=40),tick=3000,
        save=files.append,write=lambda p,v:None,args=SimpleNamespace(output=Path('/not-created')))
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<actual-entry-callbacks>','exec'),ns)
    run_schedule(alt,train=ns['train_short'],refresh=ns['refresh_pair'],finalize=ns['finalize_cycle'],checkpoint=files.append)
    assert [c[:2] for c in calls if c[0]=='train']==[('train','recovery'),('train','approach')]*2
    assert len([c for c in calls if c[0]=='refresh'])==2
    assert 'cycle_01_candidate_0032.pt' in files
    assert alt.results[-1]['result']['refresh_world_updates']==64 # not 128
    assert "if not managed_refresh and block_name=='approach'" in source
    assert "consumer_heads=('recovery','approach') if managed_refresh" in source
    assert "if actor_hash!=tensor_digest(actor.state_dict().items()):raise RuntimeError('Refresh WM/critic fitting changed actor')" in source


def test_both_adam_states_and_native_reference_survive_switch_and_restore():
    from test_supermimic_physical_actor_transaction import learner
    class Dual(nn.Module):
        def __init__(self):
            super().__init__();self.base=nn.Linear(2,2,bias=False);self.recovery=nn.Linear(2,2,bias=False);self.approach=nn.Linear(2,2,bias=False)
            for p in self.parameters():nn.init.zeros_(p)
        def forward(self,obs,sample=False):return self.base(obs)+self.recovery(obs)+self.approach(obs),None,None
    t=learner();t.actor=Dual();t.reference=copy.deepcopy(t.actor).requires_grad_(False)
    t._actor_optimizers={};t._pending_actor_optimizer_states={}
    t.native_accepted_snapshot=copy.deepcopy(t.actor.state_dict());native=copy.deepcopy(t.native_accepted_snapshot)
    for h in ('recovery','approach'):
        t.set_trainable_names([h+'.weight'],h)
        assert t.imagine_update(t.budget_states[0])['update_accepted']
    opts=copy.deepcopy({h:o.state_dict() for h,o in t._actor_optimizers.items()})
    t._actor_optimizers={};t._pending_actor_optimizer_states=opts
    before=copy.deepcopy(t.actor.state_dict())
    t.set_trainable_names(['recovery.weight'],'recovery')
    assert t.actor_opt.param_groups[0]['lr']==5e-5
    assert t.imagine_update(t.budget_states[0])['update_accepted']
    assert int(next(iter(t.actor_opt.state.values()))['step'])==2
    assert torch.equal(before['approach.weight'],t.actor.state_dict()['approach.weight'])
    assert torch.equal(before['base.weight'],t.actor.state_dict()['base.weight'])
    for k,v in native.items():assert torch.equal(v,t.native_accepted_snapshot[k])
    # Frozen head parameters still transmit gradients to inputs.
    x=torch.ones(2,2,requires_grad=True)
    assert torch.autograd.grad(t.actor.approach(x).sum(),x)[0].abs().sum()>0
    t.set_trainable_names(['approach.weight'],'approach')
    assert int(next(iter(t.actor_opt.state.values()))['step'])==1


def test_launcher_and_configs_match_pair_refresh_and_keep_native_promotion():
    import json
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/supermimic_success_guided_v2.json').read_text())
    assert c['recovery_updates']==c['approach_updates']==128 and c['head_block_size']==8
    assert c['refresh_mode']=='pair' and c['frozen_refresh']['collections_per_cycle']==16
    assert c['actor_updates_total']==c['recovery_updates']+c['approach_updates']==256
    outer=(root/'scripts/run_supermimic_success_guided_large_v1.sh').read_text()
    assert 'SM_CYCLES:-1' in outer and 'SM_RECOVERY_UPDATES:-128' in outer and 'SM_APPROACH_UPDATES:-128' in outer
    assert c['imagination_s']==.64 and c['world_window_s']==.16
    shell=(root/'scripts/run_supermimic_phase1_long_v1.sh').read_text()
    assert '--training-profile large --wm-sampling-mode promotion' in shell
    assert '--sequential-layout short-alternating' in shell
    assert 'SM_REFRESH_MODE:-pair' in shell and 'SM_TRAINING_PROFILE:-large' in shell
    assert shell.index('verify_supermimic_promotion.py')>shell.index('scripts/evaluate_supermimic_v1.py')


def test_resume_cannot_change_refresh_fitting_or_native_collection_budget():
    plan=make_plan(8,8,16,16)
    a=AlternatingSchedule(plan,execution_config=dict(refresh_world_updates=32,native_ticks_per_proposal=16))
    b=AlternatingSchedule(plan,execution_config=dict(refresh_world_updates=64,native_ticks_per_proposal=16))
    with pytest.raises(ValueError,match='configuration'):b.load_state_dict(a.state_dict())


def test_completed_research_schedule_extends_twice_without_replaying_a_block():
    def plan(cycles):
        return make_plan(128, 128, 256, 256, cycles=cycles, refresh_every_pairs=8)

    config = {'contact_gradient': 'G1'}
    schedule = AlternatingSchedule(plan(1), execution_config=config)
    for cycles in (1, 4, 8):
        if schedule.complete:
            previous = schedule.state_dict()
            schedule = AlternatingSchedule(plan(cycles), execution_config=config)
            with pytest.raises(ValueError, match='configuration'):
                schedule.load_state_dict(previous)
            corrupt = dict(previous, cursor=previous['cursor']-1)
            with pytest.raises(ValueError, match='completed prefix'):
                schedule.load_state_dict(corrupt, extend_completed=True)
            schedule.load_state_dict(previous, extend_completed=True)
            assert schedule.current['cycle'] == previous['plan'][-1]['cycle'] + 1
            assert schedule.current['head'] == 'recovery'
            assert schedule.results == previous['results']
        run_schedule(schedule, train=accepted, refresh=refreshed, finalize=lambda e: {}, checkpoint=lambda name: None)
        assert [s['accepted'] for s in schedule.head_totals().values()] == [128*cycles]*2
        assert sum(r['event']['kind']=='refresh' for r in schedule.results) == 2*cycles
