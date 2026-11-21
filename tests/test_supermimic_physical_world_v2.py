import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2, event_window_pools, sample_event_windows


ASSET=json.loads(Path('configs/go2_piper_racket_landing_model_v1.json').read_text())['asset_sha256']


def inputs():
    state=torch.zeros(4,55);state[:,2]=.3;state[:,3]=1
    feet=torch.tensor([[0.,0.,1.,1.]]).expand(4,-1).clone()
    return state,torch.zeros(4,18),torch.zeros(4,18),feet,torch.zeros(4,dtype=torch.bool),torch.zeros(4),torch.zeros(4,6)


def test_v1_migration_preserves_single_step_physical_prediction():
    old=PhysicalWorldV1(hidden=(16,16));new=PhysicalWorldV2(ASSET,hidden=(16,16),contact_hidden=(8,8))
    new.import_v1(old)
    torch.testing.assert_close(old(*inputs())[0],new(*inputs())[0],atol=0,rtol=0)
    args=list(inputs());args[3]=args[3].bool()
    probability=new.contact_probabilities(*args)
    assert (probability[:,:2]<.05).all() and (probability[:,2:]>.95).all()


def test_dynamics_fit_cannot_repurpose_contact_parameters():
    model=PhysicalWorldV2(ASSET,hidden=(16,16),contact_hidden=(8,8))
    state,target,previous,feet,present,elapsed,launch=inputs()
    for _ in range(3):
        state,logits,present=model(state,target,previous,feet,present,elapsed,launch,detach_contacts=True)
        feet=logits.sigmoid();elapsed=elapsed+.01
    state.square().mean().backward()
    assert any(p.grad is not None for p in model.body.parameters())
    assert all(p.grad is None for p in model.contact.parameters())


def test_frozen_contact_model_retains_input_gradient_for_imagination():
    model=PhysicalWorldV2(ASSET,hidden=(16,16),contact_hidden=(8,8))
    # Nonzero head weights represent a trained contact network.
    with torch.no_grad():model.contact.net[-1].weight.normal_(0,.1)
    for p in model.parameters():p.requires_grad_(False)
    args=list(inputs());args[1].requires_grad_()
    value=model.contact_probabilities(*args).sum()
    gradient=torch.autograd.grad(value,args[1])[0]
    assert gradient.isfinite().all() and gradient.abs().sum()>0


def test_contact_supervision_does_not_modify_dynamics_parameters():
    model=PhysicalWorldV2(ASSET,hidden=(16,16),contact_hidden=(8,8))
    loss=torch.nn.functional.binary_cross_entropy(model.contact_probabilities(*inputs()),torch.ones(4,4))
    loss.backward()
    assert any(p.grad is not None for p in model.contact.parameters())
    assert all(p.grad is None for p in model.body.parameters())


def test_event_sampling_has_fixed_category_share_and_no_lane_leak():
    data=dict(episode=torch.zeros(8,2,dtype=torch.long),terminated=torch.zeros(8,2,dtype=torch.bool),
        timeout=torch.zeros(8,2,dtype=torch.bool),event_state=torch.zeros(8,2,8),next_event_state=torch.zeros(8,2,8),
        foot_contact=torch.zeros(8,2,4,dtype=torch.bool),next_foot_contact=torch.ones(8,2,4,dtype=torch.bool))
    data['next_event_state'][2,:,1]=1
    pools=event_window_pools(data,torch.tensor([0]),2)
    pairs,counts=sample_event_windows(pools,100,torch.Generator().manual_seed(3))
    assert counts=={'uniform':50,'qualified_contact':25,'foot_transition':25}
    assert pairs[:,1].eq(0).all()
    # One rare contact window category retains equal quota despite the much
    # larger number of foot-transition windows.
    assert len(pools['foot_transition'])>len(pools['qualified_contact'])


def test_joint_transition_reuses_flight_without_reconnecting_the_detached_ball():
    from quadsmash.rl.supermimic_ball_residual_v1 import BallResidualV1
    from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition

    model=PhysicalWorldV2(ASSET,hidden=(16,),contact_hidden=(8,)).requires_grad_(False)
    ball=BallResidualV1(gate_hidden=(8,),response_hidden=(8,)).requires_grad_(False)
    state,target,previous,feet,present,elapsed,launch=inputs()
    state.requires_grad_()
    target.requires_grad_()
    present[2:]=True
    elapsed[1]=model.launch_s
    calls=[]
    hook=model.flight.register_forward_hook(lambda *args: calls.append(1))
    try:
        for allow_ball_gradient in (False,True):
            calls.clear()
            prediction,_,active,details=ball_transition(model,ball,state,target,previous,feet,present,elapsed,launch,
                                                        return_details=True,ball_input_grad=allow_ball_gradient)
            assert len(calls)==(2 if allow_ball_gradient else 1)
            assert active.tolist()==[False,True,True,True]
            assert details['prior_ball'].requires_grad==allow_ball_gradient
            state_grad,target_grad=torch.autograd.grad(prediction[:,:49].sum(),(state,target))
            assert state_grad.abs().sum()>0 and target_grad.abs().sum()>0
    finally:
        hook.remove()
