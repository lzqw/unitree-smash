import pytest
import torch

from quadsmash.rl.supermimic_event_hazard_v1 import event_hazard_loss, advance_event_latches


def test_only_first_event_transition_supervises_hazard():
    logits=torch.zeros(4,1,requires_grad=True)
    before=torch.tensor([[0.],[0.],[1.],[1.]])
    after=torch.tensor([[0.],[1.],[1.],[1.]])
    loss=event_hazard_loss(logits,before,after)
    loss.backward()
    torch.testing.assert_close(logits.grad,torch.tensor([[.25],[-.25],[0.],[0.]]))
    assert loss.item()==pytest.approx(.69314718)


def test_already_paid_batch_has_zero_loss_and_finite_zero_gradient():
    logits=torch.randn(10,8,requires_grad=True)
    loss=event_hazard_loss(logits,torch.ones_like(logits),torch.ones_like(logits))
    loss.backward()
    assert loss.item()==0 and logits.grad.eq(0).all()


def test_soft_survival_and_first_occurrence_mass_do_not_pay_twice():
    state=torch.zeros(1,1,dtype=torch.float64)
    hazard=torch.tensor([[.2]],dtype=torch.float64,requires_grad=True)
    logits=torch.logit(hazard)
    increments=[]
    for _ in range(20):
        following=advance_event_latches(state,logits)
        increments.append(following-state)
        state=following
    expected=1-(1-hazard)**20
    torch.testing.assert_close(state,expected)
    torch.testing.assert_close(torch.stack(increments).sum(0),state)
    state.sum().backward()
    assert hazard.grad.item()>0
    assert advance_event_latches(torch.ones_like(logits),logits).eq(1).all()


def test_reject_mismatched_event_layout():
    with pytest.raises(ValueError,match='labels'):
        event_hazard_loss(torch.zeros(2,8),torch.zeros(2,7),torch.zeros(2,8))


def test_learner_event_cycle_and_legacy_resume_rejection(tmp_path):
    from torch import nn
    from quadsmash.rl.supermimic_horizon_learner_v2 import HorizonTrainer
    from quadsmash.rl.supermimic_sequence_replay_v2 import SequenceReplay
    from quadsmash.rl.supermimic_timebase_v2 import Timebase
    class Actor(nn.Module):
        def __init__(self):
            super().__init__();self.layer=nn.Linear(4,18)
        def forward(self,x,sample=False):
            a=self.layer(x).tanh();z=a[:,:1]*0
            return a,z,z
    class World(nn.Module):
        event_dim=2
        def __init__(self):
            super().__init__();self.layer=nn.Linear(24,8)
        def forward(self,x,a,e):
            out=self.layer(torch.cat((x,a,e),-1))
            return .01*out[:,:4],out[:,4:5],out[:,5:6]-4,out[:,6:]
    replay=SequenceReplay(80,2,4,18,'cpu',event_dim=2)
    for tick in range(12):
        x=torch.full((2,4),tick*.01)
        event=torch.tensor([[float(tick>=3),float(tick>=7)]]).repeat(2,1)
        after=torch.tensor([[float(tick+1>=3),float(tick+1>=7)]]).repeat(2,1)
        replay.add(x,torch.zeros(2,18),x+.01,torch.zeros(2),torch.zeros(2),
                   torch.full((2,),tick*.01),torch.zeros(2,dtype=torch.long),
                   event_state=event,next_event_state=after)
    trainer=HorizonTrainer(Actor(),World(),nn.Linear(4,1),replay,
                           Timebase(imagination_s=.04,world_window_s=.02),
                           critic_epochs=1,critic_minibatches=1)
    trainer.fit_world(1,4)
    assert trainer.imagine_update(4)['imagine_update']
    path=tmp_path/'new.pt';trainer.save(path,{})
    trainer.load(path)
    old=torch.load(path,weights_only=False)
    del old['event_model_semantics']
    torch.save(old,tmp_path/'legacy.pt')
    parameters=[p.detach().clone() for p in trainer.actor.parameters()]
    with pytest.raises(ValueError,match='Event model semantics'):
        trainer.load(tmp_path/'legacy.pt')
    assert all(torch.equal(a,b) for a,b in zip(parameters,trainer.actor.parameters()))
