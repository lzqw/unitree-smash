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
    monkeypatch.setattr(toi,'flight_intercept',lambda state,horizon:forecast)
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
                next_state=torch.ones(n,55)*i) for i in range(16)]
    raw=dict(start={'before':{'physical':initial}},step_sequence=steps)
    win=fit.recorded_body_inputs(raw,[2,0],.01)
    torch.testing.assert_close(win['elapsed'][0],initial['elapsed_s'][[2,0]])
    torch.testing.assert_close(win['previous_action'][2],steps[1]['final_action'][[2,0]])
    torch.testing.assert_close(win['targets'][2],steps[2]['final_joint_targets'][[2,0]])
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
