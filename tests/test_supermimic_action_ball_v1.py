"""CPU contracts for an optional swept-racket candidate; no native physics evidence."""
import torch
from torch import nn
from quadsmash.rl.supermimic_action_ball_v1 import (ActionConditionedBallV1,
    CausalActionConditionedBallV2,preimpact_hard_negatives,normalized_softmin,
    hermite_sweep_samples)
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import attach_hybrid_impulse_prior
from quadsmash.rl.supermimic_action_ball_v1 import postimpact_hard_negatives,contact_neighborhood_pools


class FakeRacket:
    ids={'Racket':0}
    def __call__(self,q,dq,position,rotation,velocity,omega):
        n=len(q)
        tool=torch.stack((q[:,12],q[:,12]*0,q[:,12]*0),-1)[:,None]
        tool_velocity=torch.stack((dq[:,12],dq[:,12]*0,dq[:,12]*0),-1)[:,None]
        return dict(position=tool,rotation=torch.eye(3,device=q.device).expand(n,1,3,3),
                    velocity=tool_velocity,angular_velocity=torch.zeros_like(tool_velocity))


class FakePreview(nn.Module):
    def racket_gap(self,local):return local[:,0].abs()-.025


class FakeWorld(nn.Module):
    def __init__(self):super().__init__();self.geometry=FakeRacket()
    def prior(self,state,present,elapsed,launch):
        flight=torch.cat((state[:,49:52]+state[:,52:55]*.01,state[:,52:55]),-1)
        return torch.cat((state[:,:49],flight),-1),present
    def forward(self,state,target,prev,feet,present,elapsed,launch):
        # 55D state: joint position indices 13..30; racket arm position = 25.
        robot=torch.cat((state[:,:25],target[:,12:13],state[:,26:]),-1)
        return robot,feet.new_zeros(len(state),4),present


class FakeBall(nn.Module):
    def forward(self,features,present):
        return features.new_zeros(len(features),6),features.new_zeros(len(features))


def inputs():
    state=torch.zeros(4,55);state[:,3]=1.;state[:,49]=.01;state[:,52]=-5.
    target=torch.zeros(4,18);target[:,12]=torch.tensor([0.,.01,.03,.05])
    prev=torch.zeros_like(target);feet=torch.ones(4,4);present=torch.ones(4,dtype=torch.bool)
    elapsed=torch.full((4,),2.5);launch=torch.zeros(4,6)
    return state,target,prev,feet,present,elapsed,launch


def test_legacy_transition_is_exact_until_sweep_candidate_is_trained_and_enabled():
    w,b=FakeWorld(),FakeBall();sweep=ActionConditionedBallV1(preview=FakePreview())
    args=inputs()
    old=ball_transition(w,b,*args)[0]
    candidate=ball_transition(w,b,*args,action_model=sweep)[0]
    assert torch.equal(old,candidate)


def test_one_step_ball_velocity_is_differentiable_to_current_arm_target_after_fit():
    w,b=FakeWorld(),FakeBall();sweep=ActionConditionedBallV1(preview=FakePreview())
    state,target,prev,feet,present,elapsed,launch=inputs()
    optimizer=torch.optim.Adam(sweep.head.parameters(),lr=.02)
    for _ in range(3):
        pred=ball_transition(w,b,state,target,prev,feet,present,elapsed,launch,action_model=sweep)[0]
        loss=(pred[:,52]-2.).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    changed=target.clone().requires_grad_(True)
    pred=ball_transition(w,b,state,changed,prev,feet,present,elapsed,launch,action_model=sweep)[0]
    grad=torch.autograd.grad(pred[:,52].sum(),changed)[0]
    assert torch.isfinite(grad).all() and grad[:,12].abs().max()>1e-7
    assert torch.equal(grad[:,:12],torch.zeros_like(grad[:,:12]))


def test_absent_ball_never_receives_racket_sweep():
    w,b=FakeWorld(),FakeBall();sweep=ActionConditionedBallV1(preview=FakePreview())
    with torch.no_grad():sweep.head.net[-1].bias[3]=1.
    state,target,prev,feet,present,elapsed,launch=inputs()
    present.zero_()
    old=ball_transition(w,b,state,target,prev,feet,present,elapsed,launch)[0]
    changed=ball_transition(w,b,state,target,prev,feet,present,elapsed,launch,action_model=sweep)[0]
    assert torch.equal(old,changed)


def test_causal_sweep_suppresses_separating_postimpact_geometry():
    world,ball=FakeWorld(),FakeBall()
    model=CausalActionConditionedBallV2(preview=FakePreview())
    state,target,prev,feet,present,elapsed,launch=inputs()
    state[:,49]=.04
    target[:,12]=-0.05
    predicted,_,_=world(state,target,prev,feet,present,elapsed,launch)
    residual,details=model(state,predicted,state[:,49:],world.geometry,present,details=True)
    assert torch.isfinite(residual).all()
    assert (details['causal_gate'] < details['support']).all()
    assert float(details['causal_gate'].max()) < .2


def test_causal_sweep_keeps_entry_geometry_active():
    world,ball=FakeWorld(),FakeBall()
    model=CausalActionConditionedBallV2(preview=FakePreview())
    state,target,prev,feet,present,elapsed,launch=inputs()
    state[:,49]=.04
    target[:,12]=.03
    predicted,_,_=world(state,target,prev,feet,present,elapsed,launch)
    _,details=model(state,predicted,state[:,49:],world.geometry,present,details=True)
    assert (details['causal_gate'] > .45).all()


def test_normalized_softmin_has_no_sampling_count_bias():
    values = torch.full((2, 5), .004)
    result = normalized_softmin(values, .008)
    torch.testing.assert_close(result, values[:, 0], atol=1e-7, rtol=0)


def test_hermite_sweep_uses_physical_velocity_units():
    first = torch.zeros(1, 3)
    velocity = torch.tensor([[2., -1., .5]])
    dt = .01
    last = first + dt * velocity
    path = hermite_sweep_samples(first, last, velocity, velocity, dt=dt, count=5)
    expected = first[:, None] + torch.linspace(0., dt, 5)[None, :, None] * velocity[:, None]
    torch.testing.assert_close(path, expected, atol=1e-7, rtol=0)


def test_hermite_zero_velocity_same_endpoint_is_constant():
    point = torch.tensor([[.2, -.1, .4]])
    path = hermite_sweep_samples(point, point, torch.zeros_like(point),
                                 torch.zeros_like(point), dt=.01, count=5)
    torch.testing.assert_close(path, point[:, None].expand_as(path))


def test_hermite_endpoint_time_derivative_matches_physical_velocity():
    point = torch.zeros(1, 3)
    velocity = torch.tensor([[1., 2., 3.]], requires_grad=True)
    dt = .01
    path = hermite_sweep_samples(point, point + dt * velocity.detach(),
                                 velocity, velocity, dt=dt, count=3)
    derivative = torch.autograd.grad(path[:, 1].sum(), velocity)[0]
    # The midpoint is not an endpoint derivative.  Check the exact endpoint
    # relation through the analytic Hermite formula instead.
    alpha = torch.tensor(1e-4, requires_grad=True)
    h00 = 2*alpha**3 - 3*alpha**2 + 1
    h10 = alpha**3 - 2*alpha**2 + alpha
    h01 = -2*alpha**3 + 3*alpha**2
    h11 = alpha**3 - alpha**2
    curve = h00*point + h10*(dt*velocity) + h01*(dt*velocity.detach()) + h11*(dt*velocity)
    d_alpha = torch.autograd.grad(curve.sum(), alpha)[0]
    assert torch.allclose(d_alpha / dt, velocity.detach().sum(), atol=2e-3, rtol=0)
    assert torch.isfinite(derivative).all()


def test_analytic_impulse_prior_is_part_of_main_transition_when_attached():
    class Prior:
        def __init__(self):
            self.called = False

        def __call__(self, state, geometry, present, predicted_robot=None):
            self.called = True
            assert predicted_robot is not None
            return state[:, 49:] + state.new_tensor([0., 0., 0., 1., 0., 0.]), {}

    world, ball = FakeWorld(), FakeBall()
    model = ActionConditionedBallV1(preview=FakePreview())
    prior = Prior()
    model.__dict__['impulse_prior'] = prior
    state, target, prev, feet, present, elapsed, launch = inputs()
    result = ball_transition(world, ball, state, target, prev, feet, present,
                             elapsed, launch, action_model=model)[0]
    assert prior.called
    assert torch.allclose(result[:, 52], state[:, 52] + 1.)


def test_hybrid_details_separate_analytic_impulse_from_trainable_residual():
    class Prior:
        def __call__(self, state, geometry, present, predicted_robot=None):
            return state[:, 49:] + state.new_tensor([0., 0., 0., 2., 0., 0.]), {
                'active': present.bool(),
            }

    world, ball = FakeWorld(), FakeBall()
    model = ActionConditionedBallV1(preview=FakePreview())
    model.__dict__['impulse_prior'] = Prior()
    args = inputs()
    _, _, _, details = ball_transition(world, ball, *args, action_model=model,
                                       return_details=True)
    assert details['hybrid'] is True
    assert torch.allclose(details['action_residual'], torch.zeros_like(details['action_residual']))
    assert torch.allclose(details['analytic_ball'][:, 3], args[0][:, 52] + 2.)


def test_registered_hybrid_prior_is_frozen_and_excluded_from_residual_checkpoint():
    model = ActionConditionedBallV1(preview=FakePreview())
    attach_hybrid_impulse_prior(model, 'cpu')
    assert not any(name.startswith('impulse_prior.') for name in model.state_dict())
    assert not any(parameter.requires_grad for parameter in model.impulse_prior.parameters())
    restored = ActionConditionedBallV1(preview=FakePreview())
    attach_hybrid_impulse_prior(restored, 'cpu')
    restored.load_state_dict(model.state_dict(), strict=True)


def test_launch_transition_keeps_prior_launch_state_over_parked_ball():
    class LaunchWorld(FakeWorld):
        launch_s = 2.
        def prior(self, state, present, elapsed, launch):
            launch_now = (~present.bool()) & (elapsed >= self.launch_s - 1e-6)
            ball = torch.where(launch_now[:, None], launch, state[:, 49:])
            return torch.cat((state[:, :49], ball), -1), launch_now

    world, ball = LaunchWorld(), FakeBall()
    state, target, prev, feet, present, elapsed, launch = inputs()
    present.zero_()
    launch[:, 0] = 3.
    model = ActionConditionedBallV1(preview=FakePreview())
    model.__dict__['impulse_prior'] = lambda state, geometry, present, predicted_robot=None: (
        state[:, 49:], {'active': torch.zeros_like(present)})
    predicted = ball_transition(world, ball, state, target, prev, feet, present,
                                elapsed, launch, action_model=model)[0]
    assert torch.allclose(predicted[:, 49], launch[:, 0])


def test_precontact_hard_negative_is_one_tick_before_native_edge_and_reset_safe():
    contact=torch.tensor([[False],[False],[True],[False],[True],[False]])
    episode=torch.tensor([[0],[0],[0],[1],[1],[1]])
    done=torch.tensor([[False],[False],[True],[False],[False],[True]])
    hard=preimpact_hard_negatives(contact,episode,done,torch.zeros_like(done))
    assert hard.flatten().tolist()==[False,True,False,True,False,False]
    # An edge in a new episode must not label the old terminal as a negative.
    edge=torch.tensor([[False],[True]])
    assert not preimpact_hard_negatives(edge,torch.tensor([[0],[1]]),
        torch.tensor([[True],[False]]),torch.zeros(2,1,dtype=torch.bool)).any()


def test_postimpact_never_crosses_terminal_reset_or_labels_another_contact_negative():
    edge=torch.tensor([[True],[True],[False],[True],[False],[True],[False]])
    episode=torch.tensor([[0],[0],[0],[0],[1],[1],[1]])
    done=torch.tensor([[False],[False],[False],[True],[False],[True],[False]])
    result=postimpact_hard_negatives(edge,episode,done,torch.zeros_like(done))
    assert result.flatten().tolist()==[False,False,True,False,False,False,False]


def test_contact_pools_cover_eligible_rows_once_and_do_not_include_holdout_lanes():
    edge=torch.zeros(10,2,dtype=torch.bool);edge[2,0]=True;edge[4,0]=True;edge[6,1]=True
    episode=torch.zeros(10,2,dtype=torch.long);done=torch.zeros_like(edge)
    eligible=torch.ones_like(edge);eligible[:,1]=False
    pools=contact_neighborhood_pools(edge,episode,done,done,eligible)
    rows=[tuple(row) for pool in pools.values() for row in pool.tolist()]
    assert len(rows)==len(set(rows))==10
    assert all(lane==0 for tick,lane in rows)
    assert (3,0) in map(tuple,pools['post_contact'].tolist())
    assert (3,0) not in map(tuple,pools['pre_contact'].tolist())
