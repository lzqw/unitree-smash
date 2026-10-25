from types import SimpleNamespace
import torch
import pytest

from quadsmash.control.full_racket_residual_v4 import ResidualActionMixerV4
from quadsmash.control.full_racket_residual_reward_v4 import ResidualEventRewardV4,EVENT_WEIGHTS
from quadsmash.control.full_racket_rally_controller_v4 import FullRacketRallyControllerV4
from quadsmash.control.go2_walk_transfer import Go2WalkTransfer


def fixture():
    return torch.zeros(2,794),torch.zeros(18),torch.tensor([[-1.,1.]]*18)


def test_zero_residual_is_bitwise_base_after_fresh_reset():
    public,default,limits=fixture();m=ResidualActionMixerV4(2,'cpu',[.1]*18)
    for i in range(20):
        public[:,775]=i*.01;base=torch.rand(2,18)*2-1
        before=public.clone();result=m.action(base,torch.zeros_like(base),public,default,limits)
        assert torch.equal(result,base) and torch.equal(public,before)


def test_reward_center_override_preserves_the_actual_teacher_and_face():
    from quadsmash.rl.full_racket_residual_ppo_v4 import reward_contact_offset
    for face in [1.,-1.]:
        teacher=torch.tensor([face*.0256,0.,.1]);before=teacher.clone()
        old=reward_contact_offset(teacher,False)
        centered=reward_contact_offset(teacher,True)
        assert old is teacher and torch.equal(teacher,before)
        torch.testing.assert_close(centered,torch.tensor([face*.0256,0.,0.]))
        assert centered.data_ptr()!=teacher.data_ptr()


def test_nonzero_residual_respects_original_bounds_and_arm_slew():
    public,default,limits=fixture();m=ResidualActionMixerV4(2,'cpu',[.2]*18);last=torch.zeros(2,6)
    for i in range(40):
        public[:,775]=i*.01
        raw=torch.full((2,18),100. if i%2 else -100.)
        result=m.action(torch.zeros_like(raw),raw,public,default,limits)
        assert result.abs().max()<=1.
        assert (result[:,12:]-last).abs().max()<=.050001
        last=result[:,12:]
    for bad in [[0.]*18,[.3]*18,[.1]*17]:
        with pytest.raises(ValueError):ResidualActionMixerV4(2,'cpu',bad)


def test_selective_reset_does_not_change_other_lane_state():
    public,default,limits=fixture();m=ResidualActionMixerV4(2,'cpu',[.1]*18)
    m.action(torch.zeros(2,18),torch.ones(2,18),public,default,limits)
    saved=m.arm_rate.target[1].clone();m.reset(torch.tensor([0]))
    assert not m.nonzero_seen[0] and m.nonzero_seen[1]
    assert torch.equal(m.arm_rate.target[1],saved) and m.arm_rate.initialized[1]
    walk=object.__new__(Go2WalkTransfer);walk.last_raw=torch.ones(2,12);walk.reached=torch.ones(2,dtype=torch.bool);walk.tick=7
    walk.reset(torch.tensor([0]));assert walk.tick==7 and walk.last_raw[1].eq(1).all() and walk.reached[1]


def test_rally_selective_reset_preserves_other_episode_and_logs():
    c=object.__new__(FullRacketRallyControllerV4);called=[]
    c.teacher=SimpleNamespace(reset=lambda ids:called.append(ids.clone()))
    c.base=torch.arange(38,dtype=torch.float32);c.parameters=torch.zeros(2,38)
    c.event_time=torch.ones(2);c.planned=torch.ones(2,dtype=torch.bool)
    c.target_offset_world=torch.ones(2,3);c.logs=['kept']
    c.reset(torch.tensor([0]))
    assert torch.equal(c.parameters[0],c.base) and c.parameters[1].eq(0).all()
    assert c.event_time[1]==1 and c.planned[1] and c.target_offset_world[1].eq(1).all() and c.logs==['kept']


def test_events_paid_once_reset_per_lane_and_continuous_terms_integrate():
    r=ResidualEventRewardV4(2,'cpu');flags={k:torch.zeros(2,dtype=torch.bool) for k in EVENT_WEIGHTS}
    flags['air'][0]=True;flags['body_fault'][1]=True
    result=r.step(flags,torch.ones(2),torch.zeros(2),.01)
    torch.testing.assert_close(result,torch.tensor([40.03,-7.97]))
    torch.testing.assert_close(r.step(flags,torch.ones(2),torch.zeros(2),.01),torch.full((2,),.03))
    r.reset(torch.tensor([0]));result=r.step(flags,torch.zeros(2),torch.zeros(2),.01)
    assert result[0]==40 and result[1]==0


def test_portable_actor_export_matches_public_normalization_and_stays_cpu_only():
    import io
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    from quadsmash.rl.full_racket_residual_ppo_v4 import export_residual_policy
    public=torch.randn(3,794)
    obs=TensorDict({'policy':public},batch_size=[3])
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['policy']},18,
        actor_hidden_dims=[16],critic_hidden_dims=[16],actor_obs_normalization=True)
    policy.update_normalization(obs)
    result=export_residual_policy(dict(parameters=torch.zeros(38),asset_sha256='CPU-only'),
        policy,[.1]*18,public,dict(native_execution=False))
    assert not result['native_execution'] and not result['formal_acceptance']
    actor=torch.jit.load(io.BytesIO(result['residual_actor_jit']))
    torch.testing.assert_close(actor(public),policy.act_inference(obs),atol=1e-7,rtol=1e-7)
