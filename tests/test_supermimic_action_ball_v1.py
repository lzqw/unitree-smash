"""CPU contracts for an optional swept-racket candidate; no native physics evidence."""
import torch
from torch import nn
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1,preimpact_hard_negatives
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
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
