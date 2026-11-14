import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quadsmash.rl.supermimic_timebase_v2 import Timebase, seconds_to_steps, bptt_return, gae_returns, full_range_targets
from quadsmash.rl.supermimic_sequence_replay_v2 import SequenceReplay, NotEnoughSequenceData
from quadsmash.rl.supermimic_horizon_learner_v2 import HorizonTrainer
from quadsmash.rl.supermimic_v1 import SuperMimicWorldModel

torch.set_num_threads(1)


def test_task_seconds_not_physics_substeps():
    cfg = Timebase()
    assert cfg.imagination_steps == 64
    assert cfg.world_steps == 16
    assert cfg.manifest()['world_states'] == 17
    assert cfg.manifest()['episode_ticks'] == 800


@pytest.mark.parametrize('seconds,dt', [(0, .01), (-1, .01), (.64, 0), (.645, .01), (float('inf'), .01)])
def test_invalid_timebase_is_rejected(seconds, dt):
    with pytest.raises(ValueError):
        seconds_to_steps(seconds, dt)


def test_discount_preserves_physical_decay():
    cfg = Timebase()
    assert cfg.gamma**64 == pytest.approx(.99**32)
    assert (cfg.gamma*cfg.lam)**64 == pytest.approx((.99*.95)**32)
    step_cfg = Timebase(discount_mode='per_step')
    assert step_cfg.gamma == .99 and step_cfg.lam == .95


@pytest.mark.parametrize('seconds,steps', [(.32,32), (.64,64), (.96,96), (1.28,128), (1.60,160)])
def test_longer_task_windows_are_explicit(seconds, steps):
    assert Timebase(imagination_s=seconds).imagination_steps == steps


def test_episode_window_validation():
    with pytest.raises(ValueError):
        Timebase(imagination_s=9.)


def test_return_not_suffix_average():
    reward = torch.tensor([1.,2.,3.]).reshape(3,1,1)
    done = torch.zeros_like(reward)
    value = torch.tensor([[4.]])
    assert bptt_return(reward,done,value,.9).item() == pytest.approx(1+.9*2+.81*3+.729*4)


def test_true_termination_removes_future_and_bootstrap():
    reward = torch.tensor([1.,2.,3.]).reshape(3,1,1)
    done = torch.tensor([1.,0.,0.]).reshape(3,1,1)
    assert bptt_return(reward,done,torch.tensor([[1000.]]),.9).item() == pytest.approx(1.)


def test_soft_survival_preserves_gradients():
    reward = torch.ones(2,1,1,requires_grad=True)
    done = torch.tensor([.5,0.]).reshape(2,1,1).requires_grad_()
    value = torch.tensor([[2.]],requires_grad=True)
    ret = bptt_return(reward,done,value,.9)
    ret.sum().backward()
    assert ret.item() == pytest.approx(1+.45+.81)
    assert torch.isfinite(done.grad).all()
    assert value.grad.item() == pytest.approx(.405)


@pytest.mark.parametrize('lam',[0.,1.])
def test_gae_boundaries(lam):
    rewards = torch.tensor([1.,2.,3.]).reshape(3,1,1)
    values = torch.tensor([.1,.2,.3]).reshape(3,1,1)
    last = torch.tensor([[4.]])
    done = torch.zeros_like(values)
    actual = gae_returns(values,last,rewards,done,.9,lam)
    expected = [1+.9*.2,2+.9*.3,3+.9*4] if lam == 0 else [1+.9*2+.81*3+.729*4,2+.9*3+.81*4,3+.9*4]
    torch.testing.assert_close(actual.flatten(),torch.tensor(expected))


def make_replay(slots=40, n=2, ticks=25):
    replay = SequenceReplay(slots*n,n,4,18,'cpu')
    for tick in range(ticks):
        obs = torch.full((n,4),float(tick))
        replay.add(obs,torch.zeros(n,18),obs+1,torch.zeros(n),torch.zeros(n),torch.full((n,),tick*.01),torch.zeros(n,dtype=torch.long))
    return replay


def test_replay_exact_k_transitions():
    window = make_replay().sample_windows(8,16)
    assert len(window) == 16
    for a,b in zip(window[:-1],window[1:]):
        torch.testing.assert_close(a['next_obs'],b['obs'])


def test_replay_does_not_shorten_window():
    with pytest.raises(NotEnoughSequenceData):
        make_replay(ticks=3).sample_windows(8,16)


def test_replay_cannot_cross_done():
    r = SequenceReplay(40,1,4,18,'cpu')
    for tick in range(12):
        x = torch.full((1,4),float(tick))
        r.add(x,torch.zeros(1,18),x+1,torch.zeros(1),torch.tensor([tick==5]),torch.tensor([tick*.01]),torch.zeros(1,dtype=torch.long))
    with pytest.raises(NotEnoughSequenceData):
        r.sample_windows(8,8)


def test_replay_cannot_cross_episode_even_without_done():
    r = SequenceReplay(40,1,4,18,'cpu')
    for tick in range(12):
        x = torch.full((1,4),float(tick))
        r.add(x,torch.zeros(1,18),x+1,torch.zeros(1),torch.zeros(1),torch.tensor([tick*.01]),torch.tensor([tick//6]))
    with pytest.raises(NotEnoughSequenceData):
        r.sample_windows(8,8)


def test_replay_ring_wrap_still_contiguous():
    window = make_replay(slots=20,ticks=45).sample_windows(32,16)
    assert min(float(w['obs'].min()) for w in window) >=25
    for a,b in zip(window[:-1],window[1:]):
        torch.testing.assert_close(a['next_obs'],b['obs'])


def test_resume_breaks_physics_session_boundary():
    original = make_replay(slots=20,n=1,ticks=6)
    other = SequenceReplay(20,1,4,18,'cpu')
    other.load_state_dict(original.state_dict())
    for t in range(6):
        x=torch.zeros(1,4)
        other.add(x,torch.zeros(1,18),x,torch.zeros(1),torch.zeros(1),torch.tensor([t*.01]),torch.zeros(1,dtype=torch.long))
    with pytest.raises(NotEnoughSequenceData):
        other.sample_windows(8,8)


class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.layer=nn.Linear(4,18)
    def forward(self,x,sample=False):
        a=torch.tanh(self.layer(x)); z=a[:,:1]*0
        return a,z,z


class World(nn.Module):
    def __init__(self):
        super().__init__(); self.layer=nn.Linear(22,6)
    def forward(self,x,a):
        y=self.layer(torch.cat((x,a),-1))
        return .02*torch.tanh(y[:,:4]),y[:,4:5],y[:,5:6]-4


def make_trainer(horizon=.64,micro=4):
    cfg=Timebase(imagination_s=horizon,world_window_s=.04)
    return HorizonTrainer(Actor(),World(),nn.Linear(4,1),make_replay(),cfg,
                          actor_microbatch=micro,critic_epochs=1,critic_minibatches=2)


@pytest.mark.parametrize('horizon',[.32,.64,1.28])
def test_long_bptt_then_gae_runs_repeatedly(horizon):
    torch.manual_seed(3)
    trainer=make_trainer(horizon)
    assert trainer.fit_world(1,8)['world_updates']==1
    old_actor=[p.detach().clone() for p in trainer.actor.parameters()]
    world_before=[p.detach().clone() for p in trainer.world.parameters()]
    for _ in range(3):
        stats=trainer.imagine_update(8)
        assert stats['imagine_update']
        assert math.isfinite(stats['actor_grad_norm'])
    assert trainer.actor_updates==3 and trainer.critic_updates==6
    assert any(not torch.equal(a,b) for a,b in zip(old_actor,trainer.actor.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(world_before,trainer.world.parameters()))


def test_microbatch_accumulation_matches_full_batch_gradient():
    torch.manual_seed(10)
    a=make_trainer(.64,8); b=copy.deepcopy(a)
    start=torch.randn(8,4); elapsed=torch.zeros(8,1)
    a.actor_opt.zero_grad(); b.actor_opt.zero_grad()
    ra,_,_=a._rollout(start,elapsed); (-ra.mean()).backward()
    for i in range(0,8,2):
        rb,_,_=b._rollout(start[i:i+2],elapsed[i:i+2]); (-rb.sum()/8).backward()
    for p,q in zip(a.actor.parameters(),b.actor.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)


def test_elapsed_episode_end_masks_late_reward():
    t=make_trainer(.64)
    _,trajectory,_=t._rollout(torch.zeros(2,4),torch.tensor([[7.99],[7.98]]))
    states,rewards,dones,last,valid=trajectory
    assert valid[:,0].sum()==1 and valid[:,1].sum()==2
    assert torch.all(rewards[2:]==0)
    assert dones[0,0].item()==1


def test_full_range_action_decoder_not_small_residual():
    q=torch.tensor([[-1.,0.,1.]])
    lim=torch.tensor([[-2.,2.],[-1.,3.],[-4.,4.]])
    torch.testing.assert_close(full_range_targets(q,lim),torch.tensor([[-2.,1.,4.]]))


def test_structured_world_model_shifts_history_deterministically():
    model = SuperMimicWorldModel(794, 18, structured_history=True)
    for parameter in model.parameters():
        parameter.data.zero_()
    obs = torch.arange(2 * 794, dtype=torch.float32).reshape(2, 794)
    action = torch.zeros(2, 18)
    delta, _, _ = model(obs, action)
    next_obs = obs + delta
    history = obs[:, :759].reshape(2, 11, 69)
    expected_history = torch.cat((history[:, :1], history[:, :-1]), dim=1)
    torch.testing.assert_close(next_obs[:, :759].reshape(2, 11, 69), expected_history)
    torch.testing.assert_close(next_obs[:, 759:], obs[:, 759:])


def test_event_aware_world_model_has_explicit_latch_output():
    model = SuperMimicWorldModel(794, 18, structured_history=True, event_dim=8)
    obs = torch.zeros(2, 794)
    action = torch.zeros(2, 18)
    event = torch.zeros(2, 8)
    delta, reward, terminal, next_event_logits = model(obs, action, event)
    assert delta.shape == obs.shape
    assert reward.shape == (2, 1)
    assert terminal.shape == (2, 1)
    assert next_event_logits.shape == (2, 8)


def test_event_aware_replay_preserves_state_boundaries():
    replay = SequenceReplay(40, 1, 4, 18, 'cpu', event_dim=2)
    for tick in range(20):
        x = torch.full((1, 4), float(tick))
        event = torch.tensor([[float(tick >= 3), float(tick >= 7)]])
        replay.add(x, torch.zeros(1, 18), x + 1, torch.zeros(1), torch.zeros(1),
                   torch.tensor([tick * .01]), torch.zeros(1, dtype=torch.long),
                   event_state=event,
                   next_event_state=torch.tensor([[float(tick + 1 >= 3), float(tick + 1 >= 7)]]))
    window = replay.sample_windows(2, 8)
    assert all('event_state' in row and 'next_event_state' in row for row in window)
    for row in window:
        assert row['event_state'].shape[-1] == 2
        assert torch.all(row['next_event_state'] >= row['event_state'])


def test_policy_anchor_is_reported_and_reference_stays_frozen():
    torch.manual_seed(4)
    reference = Actor()
    trainer = HorizonTrainer(Actor(), World(), nn.Linear(4, 1), make_replay(), Timebase(imagination_s=.32, world_window_s=.04),
                             reference_actor=reference, policy_anchor_coeff=.05,
                             actor_microbatch=4, critic_epochs=1, critic_minibatches=2)
    trainer.fit_world(1, 8)
    before = [p.detach().clone() for p in reference.parameters()]
    stats = trainer.imagine_update(8)
    assert stats['imagine_update'] and stats['policy_anchor_penalty'] >= 0
    assert all(torch.equal(a, b) for a, b in zip(before, reference.parameters()))


def test_actual_environment_method_delegates_residual_to_full_range():
    # Execute the modified method, with a lightweight parent implementing the
    # original FullRacketEnvV4 formula. This is not an Isaac/PhysX runtime test.
    path=Path(__file__).parents[1]/'quadsmash/tasks/stand/supermimic_env_v1.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step')
    method.decorator_list=[]
    class Parent:
        @staticmethod
        def _pre_physics_step(obj,final):
            obj.targets=full_range_targets(final,obj._sm_residual_limits)
    scope={'torch':torch,'FullRacketEnvV4':Parent,'observe_native':lambda obj:{'policy':torch.zeros(2,794)}}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<environment-method>','exec'),scope)
    final=torch.linspace(-1,1,18).repeat(2,1)
    limits=torch.stack((torch.full((18,),-2.),torch.full((18,),2.)),-1)
    obj=SimpleNamespace(_sm_residual_controller=SimpleNamespace(
        base=SimpleNamespace(action=lambda *a:torch.zeros(2,18)),
        mixer=SimpleNamespace(action=lambda *a:final)),
        _sm_residual_default=torch.zeros(18),_sm_residual_limits=limits,joint_ids=list(range(18)),
        physical_transition_archive=None)
    scope['_pre_physics_step'](obj,torch.zeros(2,18))
    torch.testing.assert_close(obj._sm_joint_targets,2*final)


def test_checkpoint_contract_roundtrip(tmp_path):
    t=make_trainer(); t.fit_world(1,8); t.imagine_update(8)
    path=tmp_path/'state.pt'
    t.save(path,{'native_steps':50},persist_replay=True)
    other=make_trainer()
    other.load(path)
    assert other.actor_updates==1
    assert len(other.replay)==len(t.replay)
    incompatible=make_trainer(1.28)
    with pytest.raises(ValueError):
        incompatible.load(path)


def test_event_horizon_audit_missing_not_fabricated():
    from scripts.audit_supermimic_event_horizon import audit
    fields={x:x for x in ('takeoff','hit','touchdown','stable')}
    result=audit([{'takeoff':1.,'hit':1.3,'touchdown':1.6,'stable':2.5}, {'hit':1.2}],fields)
    interval=result['intervals']['takeoff_to_hit']
    assert interval['count']==1 and interval['missing']==1
    assert interval['p90_s']==pytest.approx(.3)
    assert result['intervals']['touchdown_to_stable']['fraction_within_horizon']['0.64']==0
