import copy

import pytest
import torch

from scripts.supermimic_online_forward_guard import compare_forward
from scripts.replay_supermimic_recovery_wm_updates import fit_readout_rows


def forward_scores():
    return {h:{group:dict(windows=2,body_rmse=.2,root_angular_mae=.15,
                          ball_velocity_rmse=.3,ball_forward_sign_error=.1)
               for group in ('uniform','contact-1','contact+0','contact+1')}
            for h in (1,2,4,8,16,32,64)}


def test_equal_loaded_weights_pass_online_monitoring():
    initial=forward_scores()
    assert compare_forward(initial,copy.deepcopy(initial),list(initial))['passed']


@pytest.mark.parametrize('horizon,group,metric',[(2,'uniform','root_angular_mae'),
                         (64,'uniform','ball_velocity_rmse'),(1,'contact-1','ball_velocity_rmse')])
def test_physical_drift_blocks_candidate(horizon,group,metric):
    initial=forward_scores()
    candidate=copy.deepcopy(initial)
    candidate[horizon][group][metric]+=.02
    result=compare_forward(initial,candidate,list(initial))
    assert not result['passed']


def test_missing_event_never_accepts_online_model():
    initial=forward_scores()
    candidate=copy.deepcopy(initial)
    del candidate[4]['contact-1']
    assert not compare_forward(initial,candidate,list(initial))['passed']


def test_offline_readout_update_keeps_other_outputs_and_encoder():
    model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Linear(5,4))
    before=copy.deepcopy(model.state_dict())
    fit_readout_rows(model,model[-1],slice(1,3))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    model(torch.ones(2,3)).sum().backward()
    opt.step()
    assert torch.equal(model[0].weight,before['0.weight'])
    for k,v in model[-1].named_parameters():
        assert torch.equal(v[[0,3]],before['1.'+k][[0,3]])
        assert not torch.equal(v[1:3],before['1.'+k][1:3])


@pytest.mark.parametrize('valid',[True,False])
def test_strict_native_promotion_binds_online_world(tmp_path,valid):
    import json
    from quadsmash.rl.supermimic_long_run_contract_v1 import sha256,verify_native_promotion
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
    world={'weight':torch.ones(1)}
    token=tensor_digest(world.items())
    checkpoint=tmp_path/'weights.pt'
    check=dict(passed=True,spec_sha256='spec',world_sha256=token if valid else 'different')
    torch.save(dict(config=dict(wm_forward_guard_sha256='spec'),world=world,
                    metadata=dict(online_forward_validation=check)),checkpoint)
    scores=tmp_path/'metrics.json'
    scores.write_text(json.dumps(dict(policy_execution={},evaluated_checkpoint_sha256=sha256(checkpoint),
                     num_envs=128,control_ticks=800,numerical_failures=0,native_episode_return_mean=-10,
                     ball_side_complete_except_recovery=54,complete5=28)))
    if valid:
        assert verify_native_promotion(scores,checkpoint,expected_execution={})
    else:
        with pytest.raises(ValueError,match='deployed WM'):
            verify_native_promotion(scores,checkpoint,expected_execution={})


@pytest.mark.parametrize('direction',[[1.,0.,0.],[0.,0.,1.]])
def test_friction_tangent_pair_respects_joint_limit(direction):
    from scripts.audit_supermimic_friction_load_response import independent_tangent_rows
    zero=torch.zeros(1,3)
    result=independent_tangent_rows(zero,zero,torch.tensor([3.,4.,2.]),zero[0],
                 torch.tensor([[0.,0.,1.]]),torch.tensor([2.]),1.,torch.tensor([direction]),2)
    assert torch.allclose(result,torch.tensor([-1.2,-1.6,0.]),atol=1.e-6)


def test_long_native_supervision_uses_recorded_flight_commands_and_targets(monkeypatch):
    from types import SimpleNamespace
    from quadsmash.rl import supermimic_physical_learner_v1 as learner
    # Commands in the parked lane must never contribute to the supervised loss.
    gain=torch.tensor(1.,requires_grad=True)
    def transition(body,ball,state,target,prev,feet,present,elapsed,launch,action_model):
        nxt=state.clone()
        nxt[:,52:55]=state[:,52:55]+target*gain
        return nxt,feet,present
    monkeypatch.setattr(learner,'ball_transition',transition)
    commands=torch.tensor([1.,100.]).reshape(1,2,1).expand(64,2,1)
    values=torch.zeros(64,2,55)
    window=dict(targets=commands,previous_action=commands,feet=commands,elapsed=commands,
                present=torch.tensor([True,False]).expand(64,2),state=values,next_state=values)
    adapter=SimpleNamespace(physical_model=None,ball_model=None,action_ball_model=None)
    loss=learner.native_flight_rollout_loss(adapter,window,None)
    loss.backward()
    assert loss.item()==pytest.approx((4+16+32+64)/4-.5)
    assert gain.grad.item()==pytest.approx((4+16+32+64)/4)
    gain.grad=None
    squared=learner.native_flight_rollout_loss(adapter,window,None,loss_kind='mse')
    expected=(4**2+16**2+32**2+64**2)/4
    assert squared.item()==pytest.approx(expected)
    squared.backward()
    assert gain.grad.item()==pytest.approx(2*expected)


@pytest.mark.parametrize('override,expected',[(None,1.e-6),(2.e-6,2.e-6)])
def test_resumed_world_optimizer_honors_measured_rate_and_restores_moments(override,expected):
    from scripts.train_supermimic_physical_block import restore_world_optimizer
    layer=torch.nn.Linear(2,1)
    previous=torch.optim.Adam(layer.parameters(),lr=1.e-6)
    layer(torch.ones(2)).sum().backward()
    previous.step()
    resumed=torch.optim.Adam(layer.parameters(),lr=2.e-4)
    assert restore_world_optimizer(resumed,previous.state_dict(),override)==expected
    assert resumed.param_groups[0]['lr']==expected
    assert torch.equal(resumed.state[layer.weight]['exp_avg'],previous.state[layer.weight]['exp_avg'])


def test_natural_flight_windows_exclude_reserved_lanes_and_restore_draws():
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import NaturalFlightWindows
    ticks,lanes=99,4
    data={k:torch.zeros(ticks,lanes) for k in NaturalFlightWindows.keys}
    data.update(episode=torch.ones(ticks,lanes,dtype=torch.long),
                terminated=torch.zeros(ticks,lanes,dtype=torch.bool),timeout=torch.zeros(ticks,lanes,dtype=torch.bool),
                event_state=torch.zeros(ticks,lanes,8),next_event_state=torch.zeros(ticks,lanes,8))
    data['state']=torch.arange(ticks)[:,None].expand(-1,lanes)
    data['next_event_state'][70,:,1]=1
    data['timeout'][25,0]=True
    sampler=NaturalFlightWindows(data,[1,3])
    rng=sampler.generator.get_state()
    win,selected=sampler.sample(32,0,'cpu')
    assert set(selected.tolist())=={0,2}
    assert (win['state'][0,selected==0]>25).all()
    sampler.generator.set_state(rng)
    again,ids=sampler.sample(32,0,'cpu')
    assert torch.equal(ids,selected) and torch.equal(again['state'],win['state'])


def test_frozen_refresh_validation_excludes_padding_and_other_lanes():
    from scripts.audit_supermimic_frozen_refresh_forward import held_episode_data
    ticks,lanes=800,128
    values=torch.zeros(ticks,lanes)
    keys=['state','next_state','targets','previous_action','feet','present','elapsed']
    data={k:values for k in keys}
    data.update(episode=torch.zeros_like(values,dtype=torch.long),terminated=values.bool(),timeout=values.bool(),
                valid_lengths=torch.full((lanes,),800),event_state=values.unsqueeze(-1).expand(-1,-1,8),
                next_event_state=torch.zeros(ticks,lanes,8))
    data['valid_lengths'][3]=400
    data['timeout'][data['valid_lengths']-1,torch.arange(lanes)]=True
    data['next_event_state'][120,3,1]=1
    data['next_event_state'][410,3,1]=1
    receipt=dict(complete_lanes=128,lengths=data['valid_lengths'].tolist(),episode_ids=[0]*lanes)
    selection=held_episode_data(data,receipt,[3,7])
    assert selection['contact'].nonzero().tolist()==[[120,0]]
    assert selection['episode'][400,0]==-1 and selection['episode'][400,1]==0
    data['terminated'][300,3]=True
    with pytest.raises(ValueError,match='terminal tick'):
        held_episode_data(data,receipt,[3,7])


def test_frozen_wm_replay_rejects_a_different_policy_and_missing_optimizer_step():
    from scripts.replay_supermimic_recovery_wm_updates import frozen_recovery_stage
    from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
    actor={'weight':torch.tensor([.25])}
    cfg={'wm_holdout_lanes':[1,3]}
    pre=dict(actor=actor,metadata={'policy_execution':'latch'},config=cfg,
             actor_updates=8,native_ticks=128,world_updates=32)
    post=dict(pre,native_ticks=1600,world_updates=34)
    fits=[{'world_updates':33},{'world_updates':34}]
    receipt=dict(actor_sha256=tensor_digest(actor.items()),policy_execution='latch',head='recovery',
                 consumer_heads=['recovery'],actor_unchanged=True,complete_episodes=128,shared_data_policy_id='r8',
                 policy_id='r8',start_tick=128,end_tick=1600,wm_refresh_by_head={'recovery':{'world_update_reports':fits}},
                 world_updates=2)
    prior=dict(attempt_id=9,actor_updates=8)
    stage=frozen_recovery_stage(cfg,pre,post,prior,receipt)
    assert stage['attempt_id']==9 and stage['policy_version']==8 and stage['world_update_reports']==fits
    with pytest.raises(ValueError,match='policy'):
        frozen_recovery_stage(cfg,pre,dict(post,actor={'weight':torch.tensor([.75])}),prior,receipt)
    with pytest.raises(ValueError,match='Nonconsecutive'):
        frozen_recovery_stage(cfg,pre,dict(post,world_updates=35),prior,receipt)


def test_folded_arm_position_preserves_physical_quadrature_and_velocity():
    from scripts.calibrate_supermimic_signed_arm_readout import fold_arm_integration
    from types import SimpleNamespace
    layer=torch.nn.Linear(9,59)
    model=SimpleNamespace(body=SimpleNamespace(net=[layer]),dt=.01,delta_mean=torch.linspace(-.1,.1,55),
                          delta_scale=torch.linspace(.02,2,55))
    hidden=torch.randn(11,9)
    physical=lambda: model.delta_mean+model.delta_scale*layer(hidden)[:,:55]
    before=physical().detach()
    fraction=torch.linspace(.1,.9,6)
    blend=torch.linspace(0,1,6)
    fold_arm_integration(model,fraction,blend)
    expected=before.clone()
    expected[:,25:31]=torch.lerp(before[:,25:31],model.dt*fraction*before[:,43:49],blend)
    torch.testing.assert_close(physical(),expected)
