import torch

from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical


def inputs():
    state=torch.zeros(3,55);state[:,3]=1
    state[:,49:52]=torch.tensor([6.,0.,.025])
    targets=torch.randn(3,18)*.1
    previous=torch.zeros(3,18);feet=torch.ones(3,4)
    present=torch.tensor([False,False,True]);elapsed=torch.tensor([1.,2.,3.])
    launch=torch.tensor([[7.,0.,2.,-8.,0.,3.]]).repeat(3,1)
    return state,targets,previous,feet,present,elapsed,launch


def test_model_preserves_quaternion_and_scheduled_ball():
    model=PhysicalWorldV1(hidden=(16,16))
    args=inputs();state,logits,present=model(*args)
    p=decode_physical(state)
    torch.testing.assert_close(p['root_quaternion_w'].norm(dim=-1),torch.ones(3))
    assert logits.shape==(3,4) and present.tolist()==[False,True,True]
    torch.testing.assert_close(state[0,49:],args[0][0,49:],atol=0,rtol=0)
    prior,active=model.prior(args[0],args[4],args[5],args[6])
    assert prior[1,49]>6.9 and prior[1,52]<-7.9


def test_final_target_gradients_survive_multistep_physical_rollout():
    model=PhysicalWorldV1(hidden=(16,16))
    state,target,previous,feet,present,elapsed,launch=inputs()
    target.requires_grad_()
    for _ in range(3):
        state,foot_logits,present=model(state,target,previous,feet,present,elapsed,launch)
        feet=foot_logits.sigmoid();elapsed=elapsed+.01
    gradient=torch.autograd.grad(state[:,:49].square().mean(),target)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


def test_quaternion_sign_does_not_change_state_loss():
    model=PhysicalWorldV1(hidden=(16,16))
    state=inputs()[0];other=state.clone();other[:,3:7]*=-1
    assert model.state_loss(state,other).item()==0


def test_normalization_also_respects_quaternion_double_cover():
    model=PhysicalWorldV1(hidden=(16,16))
    args=inputs();prior,_=model.prior(args[0],args[4],args[5],args[6])
    target=prior.clone();target[:,3:7]*=-1
    model.set_normalization(model.features(*args),prior,target)
    torch.testing.assert_close(model.delta_mean,torch.zeros(55),atol=0,rtol=0)


def test_event_windows_do_not_cross_reset_or_leak_validation_lanes():
    from scripts.fit_supermimic_physical_world import window_pools
    data=dict(episode=torch.tensor([[0,0],[0,0],[1,0],[1,0]]),
              terminated=torch.tensor([[False,False],[True,False],[False,False],[False,False]]),
              timeout=torch.zeros(4,2,dtype=torch.bool),
              event_state=torch.zeros(4,2,8),next_event_state=torch.zeros(4,2,8),
              foot_contact=torch.zeros(4,2,4,dtype=torch.bool),next_foot_contact=torch.zeros(4,2,4,dtype=torch.bool))
    data['next_event_state'][1,0,1]=1
    pool,rare=window_pools(data,torch.tensor([0]),2)
    assert pool.tolist()==[[0,0],[2,0]]
    assert rare.tolist()==[[0,0]]
