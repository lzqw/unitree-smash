"""Toy differentiable contracts for the actual learner method (no policy training)."""
import copy
from types import SimpleNamespace
import torch
from torch import nn
from quadsmash.rl.supermimic_physical_learner_v1 import PhysicalHorizonTrainer
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_timebase_v2 import Timebase

class Actor(nn.Module):
    def __init__(self):super().__init__();self.approach=nn.Linear(2,2,bias=False);nn.init.constant_(self.approach.weight,.05)
    def forward(self,obs,sample=False):return self.approach(obs),None,None
    def approach_gate(self,obs):return torch.ones(len(obs),1)

class State:
    def __init__(self,obs):
        self.physical=obs;self.observation=obs;self.task_memory={'rally':{'had_contact':torch.zeros(len(obs))}}
        self.executor_memory=None;self.clock_goal_context=dict(elapsed=torch.zeros(len(obs)),joint_default=torch.zeros(len(obs),2),limits=torch.tensor([[-1.,1.],[-1.,1.]]))
    def branch(self):
        x=State(self.observation.clone());x.clock_goal_context={k:v.clone() for k,v in self.clock_goal_context.items()};return x

class Adapter(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.ones(()));self.zero_reward=False;self.constant_command=False
        self.executor=SimpleNamespace(forward=self.preview)
    def critic_input(self,state):return state.observation
    def preview(self,observation,request,default,limits,memory):
        final=request*0 if self.constant_command else request
        return SimpleNamespace(final_joint_targets=final,final_normalized=final),None,None
    def forward(self,state,action):
        new=state.branch();new.observation=state.observation+action*.01*self.scale;new.physical=new.observation
        new.clock_goal_context['elapsed']+=.01
        reward=-(action-.3).square().sum(-1,keepdim=True)
        if self.zero_reward:reward=reward*0
        return new,reward,torch.ones(len(action),1),{}

def learner():
    t=PhysicalHorizonTrainer.__new__(PhysicalHorizonTrainer);t.actor=Actor();t.reference=copy.deepcopy(t.actor).requires_grad_(False)
    t.adapter=Adapter();t.critic=nn.Linear(2,1);nn.init.zeros_(t.critic.weight);nn.init.zeros_(t.critic.bias)
    t.timebase=Timebase(imagination_s=.02);t.actor_lr=5e-5;t.block_name='approach';t.reference_bank=None;t.tracking_weight=0.
    t.trainable_names={'approach.weight'};t.actor_opt=torch.optim.Adam(t.actor.parameters(),lr=5e-5)
    t.critic_opt=torch.optim.Adam(t.critic.parameters(),lr=2e-4);t.anchor_coeff=.05;t.v_max=1000.
    t.actor_updates=t.actor_attempts=t.rejected_actor_updates=t.backtracked_actor_updates=t.block_attempts=t.critic_updates=t.world_updates=0
    t.command_delta_budget=.005;t.target_delta_budget_rad=.04;t.native_reference_command_budget=None;t.native_reference_target_budget_rad=None
    t.budget_states=[State(torch.ones(2,2))]
    t.block_start_snapshot=copy.deepcopy(t.actor.state_dict());t.accepted_snapshot=copy.deepcopy(t.actor.state_dict());t.native_accepted_snapshot=copy.deepcopy(t.actor.state_dict())
    t.frozen_actor_hash=t.frozen_hash();return t

def test_real_learner_proposal_has_command_response_and_fixed_objective():
    t=learner();before=tensor_digest(t.adapter.state_dict().items())
    stats=t.imagine_update(t.budget_states[0])
    assert stats['update_accepted'] and stats['final_command_max_change']>0
    assert stats['objective_after']<=stats['objective_before']
    assert stats['task_gradient_norm']>0
    assert before==tensor_digest(t.adapter.state_dict().items())

def test_zero_task_gradient_does_not_move_adam_momentum():
    t=learner();t.actor.approach.weight.grad=torch.ones_like(t.actor.approach.weight);t.actor_opt.step()
    t.reference.load_state_dict(t.actor.state_dict());t.adapter.zero_reward=True
    before=copy.deepcopy(t.actor.state_dict());moments=copy.deepcopy(t.actor_opt.state_dict())
    stats=t.imagine_update(t.budget_states[0])
    assert not stats['update_accepted'] and stats['no_effective_gradient']
    for k,v in before.items():assert torch.equal(v,t.actor.state_dict()[k])
    for key,value in moments['state'][0].items():assert torch.equal(value,t.actor_opt.state_dict()['state'][0][key])

def test_zero_final_command_response_rolls_back_parameters_and_adam():
    t=learner();t.adapter.constant_command=True;before=copy.deepcopy(t.actor.state_dict())
    stats=t.imagine_update(t.budget_states[0])
    assert not stats['update_accepted'] and len(stats['backtrack_trials'])==5
    assert t.actor_opt.state_dict()['state']=={}
    for k,v in before.items():assert torch.equal(v,t.actor.state_dict()[k])


def test_tracking_normalizes_global_lane_time_mass_not_mean_of_time_means():
    t=learner();state=t.budget_states[0]
    t.adapter.physical_model=SimpleNamespace(geometry=None)
    original=t.adapter.forward
    def alive(s,a):
        new,reward,done,info=original(s,a)
        return new,reward,torch.zeros_like(done),info
    t.adapter.forward=alive
    class Reference:
        entries=[dict(mode='approach',terminal_verified=True,score_schema='historical-independent-complete5-v1')]
        def bind(self,*args,**kwargs):return {'matched':torch.ones(2,dtype=torch.bool)}
        def loss(self,p,elapsed,*args,**kwargs):
            first=float(elapsed[0])<.015
            weights=p.new_tensor([1.,0.] if first else [1.,1.])
            values=p[:,0]*0+(1. if first else 4.)
            return (values*weights).sum(),weights.sum(),weights
    t.reference_bank=Reference();t.tracking_weight=1.
    rollout=t._rollout_objective(state)
    assert float(rollout['tracking'])==3. # (1 + 4 + 4) / (1 + 2)
    assert rollout['tracking_effective_weight_sum']==3
    assert rollout['tracking_active_lane_steps']==3


def test_recovery_approach_recovery_bptt_and_independent_adam_cpu_contract():
    t=learner()
    class Dual(nn.Module):
        def __init__(self):
            super().__init__();self.base=nn.Linear(2,2,bias=False);self.recovery=nn.Linear(2,2,bias=False);self.approach=nn.Linear(2,2,bias=False)
            for p in self.parameters():nn.init.zeros_(p)
        def forward(self,obs,sample=False):
            action=self.base(obs)+self.recovery(obs)+self.approach(obs)
            return action,action.new_zeros(len(obs),1),action.new_zeros(len(obs),1)
    t.actor=Dual();t.reference=copy.deepcopy(t.actor).requires_grad_(False)
    t._actor_optimizers={};t._pending_actor_optimizer_states={}
    t.native_accepted_snapshot=copy.deepcopy(t.actor.state_dict())
    seen={}
    for head in ('recovery','approach','recovery'):
        t.set_trainable_names([head+'.weight'],head)
        before=copy.deepcopy(t.actor.state_dict());model_before=copy.deepcopy(t.adapter.state_dict())
        stats=t.imagine_update(t.budget_states[0])
        assert stats['update_accepted'] and stats['task_gradient_norm']>0 and stats['final_command_max_change']>0
        for name,value in t.actor.state_dict().items():
            assert (not torch.equal(value,before[name])) == (name==head+'.weight')
        for name,value in model_before.items():assert torch.equal(value,t.adapter.state_dict()[name])
        seen[head]=seen.get(head,0)+1
        assert int(next(iter(t.actor_opt.state.values()))['step'])==seen[head]
    assert seen=={'recovery':2,'approach':1}
    for name,value in t.native_accepted_snapshot.items():assert value.eq(0).all()
