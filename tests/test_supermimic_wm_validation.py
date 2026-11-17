"""Offline metrics test: labels are targets, never future rollout inputs."""
from types import SimpleNamespace
import torch
import pytest
from quadsmash.rl import supermimic_wm_validation_v1 as validation

def fixture():
    t,n=64,2
    state=torch.zeros(t,n,55);state[:,:,3]=1
    w=dict(state=state,next_state=state.clone(),targets=torch.zeros(t,n,18),
        previous_action=torch.zeros(t,n,18),feet=torch.zeros(t,n,4),next_feet=torch.zeros(t,n,4),
        present=torch.ones(t,n),elapsed=torch.arange(t)[:,None].expand(t,n)*.01,
        values=torch.zeros(t,n,2),valid=torch.ones(t,n,2),event_state=torch.zeros(t,n,8),
        next_event_state=torch.zeros(t,n,8))
    class Task:
        def forward_values(self,s,ns,u,prev,feet,elapsed,values,valid):
            return dict(numeric=values,valid_logits=torch.full_like(valid,30),hazards=torch.full((n,8),-80.))
    adapter=SimpleNamespace(physical_model=SimpleNamespace(),ball_model=None,task_model=Task())
    return adapter,w,torch.zeros(n,6)

def test_autoregression_does_not_feed_future_labels(monkeypatch):
    adapter,w,launch=fixture();seen=[]
    def transition(base,ball,s,u,prev,feet,present,elapsed,launch):
        seen.append(s.clone());return s.clone(),feet.clone(),present.clone()
    monkeypatch.setattr(validation,'ball_transition',transition)
    w['next_state'][:,:,0]=10.
    result=validation.validate_windows(adapter,w,launch)
    assert list(result)==['1','4','16','32','64']
    assert all(x[:,0].eq(0).all() for x in seen)
    assert result['64']['root_position_rmse']>0
    assert result['64']['touchdown_timing_mae_s'] is None
    assert result['64']['body_positive_count']==0
    assert result['64']['body_positive_brier'] is None

def test_native_stage_labels_do_not_call_pending_ball_side_failure():
    _,w,_=fixture();w['next_event_state'][4,0,1]=1
    groups=validation.validation_stages(w)
    assert groups['contact'].tolist()==[True,False]
    assert groups['contact_ball_side_pending'].tolist()==[True,False]
    assert not groups['recovery_failure'].any()
    w['feet'][3,0]=0;w['next_feet'][3,0]=1
    assert not validation.validation_stages(w)['touchdown'].any() # no prior jump
    w['event_state'][3,0,0]=1
    assert validation.validation_stages(w)['touchdown'].tolist()==[True,False]

def test_action_condition_diagnostic_detects_robot_response_but_no_direct_ball_response(monkeypatch):
    adapter,w,launch=fixture()
    def transition(base,ball,s,u,prev,feet,present,elapsed,launch):
        s=s.clone();s[:,13:31]+=u
        return s,feet,present
    monkeypatch.setattr(validation,'ball_transition',transition)
    result=validation.command_sensitivity(adapter,w,launch)
    assert len(result)==6
    assert all(r['robot_derivative_norm']>0 and r['ball_derivative_norm']==0 for r in result)
    with pytest.raises(ValueError,match='too short'):
        validation.validate_windows(adapter,{k:v[:2] for k,v in w.items()},launch)
