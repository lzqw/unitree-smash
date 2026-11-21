import torch

from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank, CATEGORIES
from quadsmash.rl.supermimic_toi_racket_objective import bind_intercept_reference


def test_toi_bank_keeps_lane_and_category_identity_after_reference_filter():
    bank=ActorStartBank(seed=2)
    for index in range(2):
        masks={name:torch.zeros(3,dtype=torch.bool) for name in CATEGORIES}
        masks['gate_onset'][0]=True
        masks['pre_contact'][2]=True
        bank.entries.append(dict(start={'index':index},masks=masks,stream=3,policy_version=16,native_tick=100+index))
    calls=[]
    def allowed(row):
        calls.append(row['index'])
        return torch.tensor([False,False,row['index']==1])
    row,report=bank.sample(1,eligible_lanes=allowed)
    assert row['index']==1 and row['_actor_lane_mask'].tolist()==[False,False,True]
    assert report['requested']=='gate_onset' and report['actual']=='pre_contact' and report['shortfall']==1
    assert sorted(calls)==[0,1]
    assert all(entry['masks']['gate_onset'][0] for entry in bank.entries)
    row,report=bank.sample(1,eligible_lanes=lambda r:torch.zeros(3,dtype=torch.bool))
    assert row is None and report['actual'] is None


def test_toi_start_rejects_expired_arrival_and_unmatched_reference(monkeypatch):
    from types import SimpleNamespace
    from quadsmash.rl import supermimic_toi_racket_objective as toi
    state=SimpleNamespace(physical=torch.zeros(4,55),clock_goal_context={'elapsed':torch.ones(4)})
    forecast=dict(valid=torch.tensor([True,True,False,True]),time=torch.tensor([1.1,1.,1.2,1.1]))
    monkeypatch.setattr(toi,'flight_intercept',lambda state,horizon,*,plane_offset_m=0.:forecast)
    def bind(*args,**kwargs):
        assert kwargs['require_toi'] and torch.equal(kwargs['event_time'],forecast['time'])
        return dict(matched=torch.tensor([True,True,True,False]))
    ref=bind_intercept_reference(state,SimpleNamespace(bind=bind),None,.64)
    assert ref['matched'].tolist()==[True,False,False,False]


def test_intercept_preview_checks_gate_and_precontact_latch_at_arrival():
    from types import SimpleNamespace
    from quadsmash.rl.supermimic_toi_racket_objective import preview_intercept_lanes
    def state(t):
        return SimpleNamespace(physical=torch.zeros(5,55),clock_goal_context={'elapsed':torch.full((5,),float(t))},
                               task_memory={'rally':{'had_contact':torch.tensor([0,0,int(t>0),0,0])}})
    actions=[]
    def advance(current, action, *, detach_ball):
        assert detach_ball and not torch.is_grad_enabled()
        actions.append(action)
        t=int(current.clock_goal_context['elapsed'][0])
        return state(t+1),None,torch.tensor([0.,0.,0.,float(t==0),0.]),None
    initial=state(0)
    initial.branch=lambda:state(0)
    trainer=SimpleNamespace(timebase=SimpleNamespace(imagination_steps=8,episode_s=8),adapter=advance,
        _actor_action=lambda s,sample:(torch.zeros(5),),_head_gate=lambda s:torch.tensor([1.,float(s.clock_goal_context['elapsed'][0]==0),1.,1.,1.]))
    ref={'matched':torch.ones(5,dtype=torch.bool),'intercept':{'time':torch.full((5,),2.),'valid':torch.ones(5,dtype=torch.bool)}}
    actual=preview_intercept_lanes(initial,ref,torch.tensor([True,True,True,True,False]),trainer)
    assert actual.tolist()==[True,False,False,False,False]
    assert len(actions)==2


def test_native_body_source_uses_executed_command_history_and_checks_resets(monkeypatch):
    import pytest
    from scripts import supermimic_paired_body_candidate as fit
    n=3
    initial=dict(foot_contact=torch.ones(n,4),ball_present=torch.ones(n),elapsed_s=torch.tensor([1.,1.1,1.2]),
                 final_action=torch.tensor([[0.],[1.],[2.]]))
    monkeypatch.setattr(fit,'encode_physical',lambda _:torch.zeros(n,55))
    steps=[dict(done=torch.zeros(n,dtype=torch.bool),elapsed_s=initial['elapsed_s']+.01*(i+1),
                final_action=initial['final_action']+100*i,final_joint_targets=initial['final_action']+5*i,
                foot_contact=torch.full((n,4),float(i)),ball_present=torch.full((n,),float(i==2)),
                next_state=torch.ones(n,55)*i) for i in range(16)]
    raw=dict(start={'before':{'physical':initial}},step_sequence=steps)
    win=fit.recorded_body_inputs(raw,[2,0],.01)
    torch.testing.assert_close(win['elapsed'][0],initial['elapsed_s'][[2,0]])
    torch.testing.assert_close(win['previous_action'][2],steps[1]['final_action'][[2,0]])
    torch.testing.assert_close(win['targets'][2],steps[2]['final_joint_targets'][[2,0]])
    for field, source in [('state','next_state'),('feet','foot_contact'),('present','ball_present')]:
        torch.testing.assert_close(win[field][3],steps[2][source][[2,0]])
    steps[3]['done'][0]=True
    with pytest.raises(ValueError,match='uninterrupted'):
        fit.recorded_body_inputs(raw,[2,0],.01)


def test_signed_arm_loss_learns_opposing_commands_without_penalizing_common_offset(monkeypatch):
    from types import SimpleNamespace
    from scripts import supermimic_paired_body_candidate as fit

    rows = [*range(25,31), *range(43,49)]
    monkeypatch.setattr(fit, 'CHECKS', (1,))
    monkeypatch.setattr(fit, 'signed_rollout', lambda world,ball,window,launch: iter([window['prediction']]))
    monkeypatch.setattr(fit, 'robot_loss', lambda state,truth: state.new_zeros(()))
    monkeypatch.setattr(fit, 'fields', lambda state,geometry: state[:,:12])
    truth = torch.zeros(4,55)
    truth[2:,rows] = 1
    prediction = truth.clone()
    prediction[:,rows] += 2
    world = SimpleNamespace(geometry=None)

    def evaluate(value):
        return fit.paired_terms(world, None, {'prediction':value, 'next_state':truth[None]},
                                torch.ones(4,6), {1:torch.ones(12)}, training=False,
                                arm_rows=rows, arm_scales={1:torch.ones(12)})

    loss,scores = evaluate(prediction)
    assert loss == 0 and scores[1]['arm_position_signed_rmse_rad'] == 0
    prediction[2,rows[1]] -= .5
    prediction.requires_grad_()
    loss,scores = evaluate(prediction)
    loss.backward()
    assert loss > 0 and scores[1]['arm_position_signed_rmse_rad'] > 0
    assert scores[1]['arm_velocity_signed_rmse_rad_s'] == 0
    assert prediction.grad[2,rows[1]] < 0 and prediction.grad[0,rows[1]] > 0
    assert prediction.grad.count_nonzero() == 2


def test_pool_toi_starts_validates_arrivals_and_deduplicates_same_lane_episode():
    from quadsmash.rl.supermimic_actor_start_bank_v1 import pool_toi_starts

    def row(t,episode,mask):
        return dict(before=dict(elapsed_tick=torch.full((3,),t),episode_index=torch.tensor(episode)),
                    _actor_lane_mask=torch.tensor(mask,dtype=torch.bool))
    first=row(10,[1,1,1],[1,0,0])
    duplicate=row(11,[1,1,1],[1,0,1])
    later=row(20,[2,1,1],[1,1,1])
    rows,layout=pool_toi_starts(first,[(duplicate,torch.tensor([True,True,False])),(later,None)],
                                lambda r:r['_actor_lane_mask'],3,1)
    assert [s['before'] is orig['before'] for s,orig in zip(rows,(first,later),strict=True)]==[True,True]
    assert [r['lane_episode_tick'] for r in layout]==[[[0,1,10]],[[0,2,20],[1,1,20]]]
    assert first['_actor_lane_mask'].tolist()==[True,False,False]


def test_pooled_toi_backtracking_preserves_each_arrival_and_global_weight():
    from types import SimpleNamespace
    from quadsmash.rl.supermimic_recovery_batches import pooled_objective

    actor=torch.nn.Linear(1,1,bias=False)
    trainer=SimpleNamespace(block_name='approach',approach_objective='toi_racket_only',actor=actor,
                            anchor_coeff=0.,tracking_coefficient=lambda head:1.)
    def objective(n,effective_weights=None,baseline_toi_weights=None,tracking_parameters=()):
        w=torch.full((1,n),float(n))
        valid=torch.ones(1,n,dtype=torch.bool)
        return dict(gate_lanes=n,gate_steps=n,tracking_steps=1,tracking_active_lane_steps=n,
                    tracking_effective_weight_sum=float(w.sum()),tracking_nonzero_gradient_lanes=n,matched_references=n,
                    task=actor.weight.sum()*7,tracking=actor.weight.sum()*n,anchor=actor.weight.sum()*0,
                    reference_coverage=dict(selected_start_lanes=n,gate_active_selected_start_lanes=n),
                    effective_weights=valid[0],ret=w.reshape(-1,1),toi_weights=w,
                    toi_coverage_preserved=baseline_toi_weights is None or torch.equal(w,baseline_toi_weights),
                    critic_data=(w[:,:,None],w[:,:,None],w[:,:,None]*0,w.T,valid))
    trainer._rollout_objective=objective
    a=pooled_objective(trainer,[1,2])
    a['loss'].backward()
    torch.testing.assert_close(actor.weight.grad,torch.full_like(actor.weight,1.8))
    with torch.no_grad():
        b=pooled_objective(trainer,[1,2],a['effective_weights'],a['toi_weights'])
        assert b['toi_coverage_preserved']
        a['toi_weights'][1][0,0]=0
        assert not pooled_objective(trainer,[1,2],a['effective_weights'],a['toi_weights'])['toi_coverage_preserved']


def test_drive_fit_uses_signed_and_absolute_motion_and_tolerates_unexcited_joint():
    from scripts.supermimic_arm_drive_audit import fit_drive
    rng=torch.Generator().manual_seed(5)
    features=torch.randn(16,12,6,2,generator=rng)
    features[:,:,4]=0
    coef=torch.randn(6,2,generator=rng)
    errors=(features*coef).sum(-1)
    actual,rank=fit_drive(features,errors)
    torch.testing.assert_close((features*actual).sum(-1),errors)
    assert rank==[2,2,2,2,0,2]


def test_tangent_probe_direction_uses_an_independent_executed_arm_axis():
    from scripts.declare_supermimic_tangent_probe import tangent_directions
    jac=torch.tensor([[[5.,0.,0.,0.,0.,0.],[0.,4.,0.,0.,0.,0.],[0.,0.,3.,0.,0.,0.]]])
    normal=torch.tensor([[1.,0.,0.]])
    previous=torch.tensor([[0.,1.,0.,0.,0.,0.]])
    d=tangent_directions(jac,normal,previous)
    torch.testing.assert_close(d,torch.tensor([[0.,0.,1.,0.,0.,0.]],dtype=torch.float64))


def test_body_trajectory_objective_preserves_source_weights_when_windows_repeat():
    from scripts.supermimic_arm_rollout_calibration import calibration_objective
    panels={(s,4,'center'):torch.tensor([.5,1.,2.]) for s in ('paired','natural','racket','recovery')}
    ref=calibration_objective(panels,panels).square().sum()
    key=('natural',4,'center')
    with_repeats=dict(panels)
    with_repeats[key]=panels[key].repeat(5)
    torch.testing.assert_close(ref,torch.tensor(1.))
    torch.testing.assert_close(calibration_objective(with_repeats,with_repeats).square().sum(),ref)


def test_alternative_arm_probe_rejects_repeated_direction_or_changed_prefix():
    import pytest
    from scripts.compare_supermimic_probe_directions import verify_alternate_directions

    first = [0.] * 12 + [1., 0., 0., 0., 0., 0.]
    second = [0.] * 12 + [0., 1., 0., 0., 0., 0.]
    old = dict(lanes=[9], tick=278, request_directions=[first], direction_seed=83)
    alternate = dict(lanes=[9], tick=278, request_directions=[second],
                     direction_design='training_prefix_tangent_orthogonal', source_direction_seed=83)
    verify_alternate_directions(old, alternate)
    for change in (dict(request_directions=[first]), dict(tick=277), dict(source_direction_seed=84)):
        with pytest.raises(ValueError):
            verify_alternate_directions(old, dict(alternate, **change))

def test_contact_plane_is_fitted_only_to_selected_training_lanes():
    import pytest
    from quadsmash.rl.supermimic_toi_racket_objective import calibrated_plane_offset
    cases=[dict(lane=i,qualified=True,valid=True,native_minus_plane_m=x) for i,x in [(0,.04),(3,.06)]]
    fit=dict(schema='training-native-contact-target-plane-v1',checkpoint_sha256='weights',fit_samples=2,plane_offset_m=.05,
             training=[dict(seed=7,lanes=[0,3],cases=cases)])
    assert calibrated_plane_offset(fit,[2],'weights')==.05
    for held,key in [([3],'weights'),([2],'new weights')]:
        with pytest.raises(ValueError):
            calibrated_plane_offset(fit,held,key)
    with pytest.raises(ValueError):
        calibrated_plane_offset(dict(fit,plane_offset_m=.07),[2],'weights')


def test_advanced_intercept_recomputes_both_position_and_time_for_original_valid_lanes(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from quadsmash.rl import supermimic_toi_racket_objective as toi
    controller=SimpleNamespace(parameters=torch.ones(2,25),physics={},height_intercept={})
    state=SimpleNamespace(observation=torch.zeros(2,794),executor_memory=SimpleNamespace(reference_controller=controller),
        clock_goal_context=dict(elapsed=torch.tensor([2.,2.]),ball_present=torch.ones(2,dtype=torch.bool)))
    monkeypatch.setattr(toi,'predict_height_intercept',lambda *args,**kw:dict(valid=np.array([True,False]),
        position_m=np.array([[1.,2.,3.],[1.,2.,3.]]),arrival_time_s=np.array([.2,.2])))
    def shifted(obs,position,normal,plane,physics,**kw):
        np.testing.assert_allclose(plane,1.05)
        return dict(valid=np.array([True,True]),position_m=np.array([[1.05,1.9,3.1],[1.05,1.9,3.1]]),
                    arrival_time_s=np.array([.19,.19]))
    monkeypatch.setattr(toi,'predict_initial_intercept',shifted)
    result=toi.flight_intercept(state,.64,plane_offset_m=.05)
    torch.testing.assert_close(result['position'],torch.tensor([[1.05,1.9,3.1],[0.,0.,0.]]))
    torch.testing.assert_close(result['time'],torch.tensor([2.19,torch.inf]))


def test_compact_preview_maps_sparse_results_to_original_lanes(monkeypatch):
    from types import SimpleNamespace
    from quadsmash.rl import supermimic_toi_racket_objective as objective
    from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1

    allowed=torch.zeros(128,dtype=torch.bool)
    allowed[[8,69,81]]=True
    present=allowed.clone()
    present[69]=False
    present[2]=True
    row=dict(observation=torch.arange(128)[:,None], before=dict(physical=dict(ball_present=present)),
             executor_before=dict(tensors={},values={}))
    def state(record, *, executor_template, launch, device):
        assert record['observation'].flatten().tolist()==[8,81]
        assert launch[:,0].tolist()==[8,81]
        return record['observation']
    monkeypatch.setattr(PhysicalImaginationStateV1,'from_record',staticmethod(state))
    monkeypatch.setattr(objective,'bind_intercept_reference',lambda *a,**k: {'matched':torch.tensor([True,True])})
    monkeypatch.setattr(objective,'preview_intercept_lanes',lambda *a:torch.tensor([False,True]))
    trainer=SimpleNamespace(timebase=SimpleNamespace(control_dt_s=.01,imagination_steps=64),
        adapter=SimpleNamespace(physical_model=SimpleNamespace(geometry=None)),reference_bank=None,toi_plane_offset_m=0.)
    launch=torch.arange(128)[:,None].expand(-1,6)
    result=objective.compact_intercept_preview(row,allowed,launch,None,trainer,'cpu')
    assert torch.where(result)[0].tolist()==[81]
    assert not objective.compact_intercept_preview(row,torch.zeros_like(allowed),launch,None,None,'cpu').any()


def test_bank_preview_union_preserves_each_category_and_shared_record():
    bank = ActorStartBank(seed=1)
    row = {}
    first = {name:torch.zeros(5,dtype=torch.bool) for name in CATEGORIES}
    second = {name:mask.clone() for name,mask in first.items()}
    first['gate_onset'][1] = True
    second['failure_pre'][3] = True
    bank.entries = [dict(start=row,masks=first),dict(start=row,masks=second)]
    mask=bank.candidate_lanes_by_start()[id(row)]
    assert mask.tolist() == [False,True,False,True,False]
    assert first['gate_onset'].tolist() == [False,True,False,False,False]


def test_extension_interleaves_same_quota_across_adjacent_blocks_and_resume():
    from collections import Counter
    from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank, SCHEDULE
    bank = ActorStartBank()
    requested = [bank.sample(draw, interleaved=True)[1]['requested'] for draw in range(129, 149)]
    assert Counter(requested) == Counter(SCHEDULE)
    assert all(a != b for a, b in zip(requested, requested[1:] + requested[:1]))
    assert all(len(set(requested[i:i+8])) == 4 for i in range(13))
    assert [bank.sample(i)[1]['requested'] for i in range(1, 21)] == list(SCHEDULE)
