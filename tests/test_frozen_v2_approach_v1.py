"""Policy invariants required before another expensive native training run."""
import pytest
import torch
from tensordict import TensorDict

from quadsmash.rl.phase_gated_recovery_v1 import (
    PhaseGatedRecoveryActorCriticV1, PhaseGatedApproachRecoveryActorCriticV1,
)
from quadsmash.rl.frozen_v2_approach_v1 import FrozenV2ApproachActorCriticV1, PublicAerialProximityGate


def observations(count=30):
    raw=torch.zeros(count,794)
    for frame in range(11):
        raw[:,frame*69+67:frame*69+69]=1.
    raw[:,61]=.1
    raw[:,770:775]=torch.nn.functional.one_hot(torch.arange(count)%5,5).float()
    return TensorDict({'policy':raw},batch_size=[count])


def make(kind, **overrides):
    kwargs=dict(actor_hidden_dims=[256,128,64],critic_hidden_dims=[256,128,64],activation='elu',
        init_noise_std=.1,noise_std_type='log',actor_obs_normalization=True,critic_obs_normalization=True,
        landing_exploration_scale=.15,airborne_hit_exploration_scale=.15,
        recovery_head_hidden_dims=[128,64],recovery_head_output_scale=.37)
    kwargs.update(overrides)
    return kind(observations(),{'policy':['policy'],'critic':['policy']},18,**kwargs)


@pytest.mark.parametrize('kind,extra',[
    (PhaseGatedRecoveryActorCriticV1,{}),
    (PhaseGatedApproachRecoveryActorCriticV1,dict(approach_phase_start_index=772,
        approach_phase_end_index=773,approach_head_output_scale=.2)),
    (FrozenV2ApproachActorCriticV1,dict(approach_head_output_scale=.2)),
])
def test_nonzero_nonunit_heads_export_every_phase(kind,extra):
    p=make(kind,**extra)
    with torch.no_grad():
        p.recovery_head[-1].bias.fill_(.4)
        if hasattr(p,'approach_head'):
            p.approach_head[-1].bias.fill_(.6)
    obs=observations()
    script=torch.jit.script(p.export_deterministic_actor())
    with torch.no_grad():
        torch.testing.assert_close(script(obs['policy']),p.act_inference(obs),atol=1e-7,rtol=1e-7)


def test_phase772_export_does_not_invent_takeoff_correction():
    p=make(PhaseGatedApproachRecoveryActorCriticV1,approach_phase_start_index=772,
        approach_phase_end_index=773,approach_head_output_scale=.2)
    obs=observations()
    with torch.no_grad():
        nominal=p.act_inference(obs).clone()
        p.approach_head[-1].bias.fill_(1.)
        actual=torch.jit.script(p.export_deterministic_actor())(obs['policy'])
    changed=(actual-nominal).abs().amax(dim=1)>1e-7
    assert torch.equal(changed,obs['policy'][:,772]>.5)


def test_hit_flag_does_not_cut_proximity_gate():
    gate=PublicAerialProximityGate()
    before=observations(2)['policy']
    before[:,770:775]=0.;before[:,772]=1.
    after=before.clone();after[:,772]=0.;after[:,773]=1.
    assert torch.equal(gate(before),gate(after))
    frames=before[:1].expand(20,-1).clone()
    frames[:,61]=torch.linspace(.2,4.,20)
    value=gate(frames).squeeze()
    assert (value[1:]<=value[:-1]).all()
    assert float(value[0])==1. and float(value[-1])<1e-6


def test_ground_handoff_decays_with_actual_contact_history():
    gate=PublicAerialProximityGate(history_frames=6)
    raw=observations(7)['policy']
    for row in range(7):
        for frame in range(row):
            raw[row,69*frame+57:69*frame+61]=1.
    weights=gate(raw).squeeze()
    torch.testing.assert_close(weights,torch.linspace(1.,0.,7))


def test_full_old_policy_and_normalizer_stay_frozen_after_optimizer_steps():
    p=make(FrozenV2ApproachActorCriticV1)
    # The inherited recovery must be nonzero, so dropping/freeing it is caught.
    with torch.no_grad():p.recovery_head[-1].bias.fill_(.25)
    p.capture_frozen_reference();p.set_recovery_trainable(True)
    assert all(not v.requires_grad for v in p.recovery_head.parameters())
    trainable={id(x) for x in p.parameters() if x.requires_grad}
    assert all(id(x) in trainable for x in p.approach_head.parameters())
    assert not any(id(x) in trainable for x in p.actor.parameters())
    obs=observations()
    optimizer=torch.optim.Adam(p.parameters(),lr=.001)
    for _ in range(3):
        p.update_normalization(obs)
        optimizer.zero_grad();(p.act_inference(obs)-.7).square().mean().backward();optimizer.step()
    assert p.assert_frozen_reference()['all_bitwise_unchanged']
    assert int(p.actor_obs_normalizer.count)==0
    assert int(p.critic_obs_normalizer.count)==90
    p.set_recovery_trainable(False)
    assert not any(v.requires_grad for v in p.actor_update_parameters())


def test_new_zero_head_preserves_the_entire_nonzero_v2_mapping():
    old=make(PhaseGatedRecoveryActorCriticV1)
    new=make(FrozenV2ApproachActorCriticV1)
    with torch.no_grad():old.recovery_head[-1].bias.fill_(.5)
    state=new.state_dict()
    state.update(old.state_dict());new.load_state_dict(state,strict=True)
    obs=observations()
    with torch.no_grad():
        assert torch.equal(old.act_inference(obs),new.act_inference(obs))
        assert torch.equal(torch.jit.script(old.export_deterministic_actor())(obs['policy']),
                           torch.jit.script(new.export_deterministic_actor())(obs['policy']))


def test_gaussian_likelihood_and_sampling_use_the_same_local_variance():
    p=make(FrozenV2ApproachActorCriticV1)
    raw=observations()['policy']
    for frame in range(11):raw[:10,frame*69+57:frame*69+61]=1.
    obs=TensorDict({'policy':raw},batch_size=[len(raw)])
    action=p.act(obs)
    expected=p.action_std_for_observations(obs)
    torch.testing.assert_close(p.action_std,expected)
    torch.testing.assert_close(expected[:10],torch.full_like(expected[:10],.0001))
    assert (expected[10:]>.0001).all()
    assert torch.isfinite(p.get_actions_log_prob(action)).all()
