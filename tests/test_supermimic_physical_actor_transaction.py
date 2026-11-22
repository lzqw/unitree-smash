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
    for _ in range(2):
        stats=t.imagine_update(t.budget_states[0])
        assert stats['update_accepted'] and stats['final_command_max_change']>0
        assert stats['objective_after']<=stats['objective_before']
        assert stats['task_gradient_norm']>0
        assert before==tensor_digest(t.adapter.state_dict().items())
        budget=stats['budget_report']
        assert budget==stats['backtrack_trials'][-1]['budget_report']
        assert budget['relative_last_accepted']['final_joint_target_delta_rad']==stats['final_command_max_change']
    assert budget['relative_block_start']['final_joint_target_delta_rad']>stats['final_command_max_change']

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


def test_budget_rejected_step_skips_rescoring_but_scores_the_committed_step_or_last_rejection():
    for budget, accepted, expected_alpha in ((.000075, True, .5), (0., False, .0625)):
        t = learner()
        t.command_delta_budget = budget
        initial = copy.deepcopy(t.actor.state_dict())
        score_calls = []
        objective = t._rollout_objective
        def counted(*args, **kwargs):
            if kwargs.get('collect_critic_data') is False:
                score_calls.append(t.actor.approach.weight.detach().clone())
            return objective(*args, **kwargs)
        t._rollout_objective = counted
        stats = t.imagine_update(t.budget_states[0])
        trials = stats['backtrack_trials']
        assert stats['update_accepted'] is accepted
        assert len(score_calls) == 1
        assert all(r['scoring_skipped'] == 'command_or_target_budget' and
                   not r['budget_ok'] and r['loss'] is None for r in trials[:-1])
        assert trials[-1]['alpha'] == expected_alpha and trials[-1]['loss'] is not None
        if accepted:
            assert torch.equal(t.actor.approach.weight, score_calls[0])
            assert stats['objective_after'] == trials[-1]['loss']
        else:
            assert torch.equal(t.actor.approach.weight, initial['approach.weight'])
            assert t.actor_opt.state_dict()['state'] == {}


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


def test_pooled_backtrack_keeps_bootstrap_and_skips_per_tick_critic_features():
    t=learner()
    t.block_name='recovery'
    t._head_gate=lambda state: torch.ones(len(state.physical))
    original=t.adapter.forward
    def continuing_step(state, action):
        new,reward,done,details=original(state,action)
        return new,reward,torch.zeros_like(done),details
    t.adapter.forward=continuing_step
    t.critic.bias.data.fill_(1.5)
    starts=(State(torch.ones(2,2)),State(torch.full((3,2),.3)))
    calls=[]
    def features(state):
        calls.append(len(state.physical))
        return state.observation
    t.adapter.critic_input=features
    with torch.no_grad():
        baseline=t._rollout_objective(starts)
        full=t._rollout_objective(starts,effective_weights=baseline['effective_weights'])
        calls.clear()
        score=t._rollout_objective(starts,effective_weights=baseline['effective_weights'],collect_critic_data=False)
    assert calls == [2,3]
    assert score['critic_data'] is None and full['critic_data'][0].shape[:2]==(2,5)
    for key in ('loss','task','tracking','anchor','ret'):
        torch.testing.assert_close(score[key],full[key],rtol=0,atol=0)
    assert score['toi_coverage_preserved']==full['toi_coverage_preserved']


def test_backtrack_records_uphill_adam_direction_against_the_unclipped_objective():
    t=learner()
    for param in t.actor.parameters():
        t.actor_opt.state[param]=dict(step=torch.tensor(10.),exp_avg=torch.ones_like(param),
                                      exp_avg_sq=torch.ones_like(param))
    stats=t.imagine_update(t.budget_states[0])
    assert not stats['update_accepted'] and stats['adam_proposal_loss_slope'] > 0
    base=stats['objective_terms_before']
    assert abs(base['task']+t.anchor_coeff*base['anchor']-stats['objective_before'])<1e-6
    for trial in stats['backtrack_trials']:
        assert trial['budget_ok'] and trial['toi_coverage_preserved']
        assert trial['loss'] > stats['objective_before']
        terms=trial['objective_terms']
        assert abs(terms['task']+t.anchor_coeff*terms['anchor']-trial['loss'])<1e-6


def test_restarted_uphill_momentum_advances_once_and_rejected_trial_restores_both_moments():
    for force_rejection in (False,True):
        t=learner()
        t.restart_uphill_approach_momentum=True
        t.adapter.constant_command=force_rejection
        p=next(t.actor.parameters())
        t.actor_opt.state[p]=dict(step=torch.tensor(10.),exp_avg=torch.ones_like(p),exp_avg_sq=torch.ones_like(p))
        old=copy.deepcopy(t.actor_opt.state_dict())
        stats=t.imagine_update(t.budget_states[0])
        assert stats['adam_proposal_loss_slope']>0 and stats['restarted_adam_proposal_loss_slope']<0
        assert stats['update_accepted'] != force_rejection
        state=t.actor_opt.state[p]
        if force_rejection:
            for k,v in old['state'][0].items():
                assert torch.equal(state[k],v)
        else:
            assert state['step']==11 and (state['exp_avg']<0).all()
            beta2=t.actor_opt.param_groups[0]['betas'][1]
            torch.testing.assert_close(state['exp_avg_sq'],beta2*old['state'][0]['exp_avg_sq']+(1-beta2)*p.grad.square())
