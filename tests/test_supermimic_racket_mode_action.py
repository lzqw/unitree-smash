"""Mixture semantics only: synthetic tests are not physical admission."""
import pytest
import torch
from torch import nn
from quadsmash.rl.supermimic_action_ball_v1 import (
    ActionConditionedBallV1, RacketModeActionBallV2,
    CausalRacketModeActionBallV3, TemporalCausalRacketModeTargetActionBallV8,
    TemporalCausalRacketModeTargetActionBallV9, migrate_temporal_v8,
    TemporalCausalRacketModeTargetLinearActionBallV10, migrate_temporal_v9,
    migrate_temporal_v10, TemporalCausalRacketModeNativeResponseBallV11,
    migrate_temporal_v11, TemporalCausalRacketModeNativeResponseBallV12,
    migrate_temporal_v12, TemporalCausalRacketModeImpulseResidualBallV13,
    migrate_temporal_v13, action_ball_for_schema)
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from test_supermimic_action_ball_v1 import FakeWorld, FakePreview, FakeBall, inputs


class ModeBall(nn.Module):
    def __init__(self, probability, mode='racket'):
        super().__init__()
        self.probability = probability
        self.mode = mode

    def components(self, features):
        p = self.probability.expand(len(features))
        zero = torch.zeros_like(p)
        if self.mode == 'frame':
            return torch.stack((1-p, zero, zero, zero, p), -1), features.new_zeros(len(features),4,6)
        return torch.stack((1-p, p, zero, zero, zero), -1), features.new_zeros(len(features),4,6)


def correction_model():
    model = RacketModeActionBallV2(preview=FakePreview())
    with torch.no_grad():
        model.head.net[-1].bias[3] = 1.
    return model


def test_free_mode_suppresses_extra_impulse_without_changing_robot_command():
    world = FakeWorld();args = inputs();model = correction_model()
    old = ball_transition(world, FakeBall(), *args)[0]
    new = ball_transition(world, ModeBall(torch.tensor(0.)), *args, action_model=model)[0]
    assert torch.equal(new, old)


def test_mode_probability_scales_impulse_without_probability_renormalization():
    world = FakeWorld();args = inputs();model = correction_model()
    free = ball_transition(world, FakeBall(), *args)[0]
    full = ball_transition(world, ModeBall(torch.tensor(1.)), *args, action_model=model)[0]
    half = ball_transition(world, ModeBall(torch.tensor(.5)), *args, action_model=model)[0]
    assert torch.allclose(half[:,49:]-free[:,49:], .5*(full[:,49:]-free[:,49:]))
    assert torch.equal(half[:,:49],full[:,:49])


def test_joint_target_and_predicted_mode_both_keep_gradients():
    world=FakeWorld();state,target,*other=inputs()
    target.requires_grad_(True);probability=torch.tensor(.6,requires_grad=True)
    predicted=ball_transition(world,ModeBall(probability),state,target,*other,
                              action_model=correction_model())[0]
    command_grad,mode_grad=torch.autograd.grad(predicted[:,52].sum(),(target,probability))
    assert torch.isfinite(command_grad).all() and command_grad[:,12].abs().max()>1e-7
    assert torch.isfinite(mode_grad) and mode_grad.abs()>1e-7


def test_frame_aware_candidate_uses_frame_probability_without_changing_v4():
    world=FakeWorld();args=inputs();model=correction_model()
    model.requires_frame_probability=True
    frame=ball_transition(world,ModeBall(torch.tensor(1.),mode='frame'),*args,action_model=model)[0]
    free=ball_transition(world,ModeBall(torch.tensor(0.),mode='frame'),*args,action_model=model)[0]
    assert (frame[:,49:]-free[:,49:]).abs().sum()>0
    legacy=ball_transition(world,ModeBall(torch.tensor(1.),mode='frame'),*args,action_model=correction_model())[0]
    assert torch.equal(legacy,free)


def test_checkpoint_schema_dispatch_preserves_old_computation_and_strict_shapes():
    old=ActionConditionedBallV1(preview=FakePreview())
    restored=action_ball_for_schema(old.schema,preview=FakePreview())
    restored.load_state_dict(old.state_dict(),strict=True)
    new=action_ball_for_schema(RacketModeActionBallV2.schema,preview=FakePreview())
    new.load_state_dict(old.state_dict(),strict=True)
    assert type(restored) is ActionConditionedBallV1
    assert type(new) is RacketModeActionBallV2
    assert old.schema != new.schema
    assert torch.equal(ball_transition(FakeWorld(),FakeBall(),*inputs(),action_model=old)[0],
                       ball_transition(FakeWorld(),FakeBall(),*inputs(),action_model=restored)[0])
    with pytest.raises(ValueError,match='Unknown'):
        action_ball_for_schema('unrecognized')


def test_no_silent_fallback_when_ball_model_has_no_mode_outputs():
    with pytest.raises(ValueError,match='object-separated'):
        ball_transition(FakeWorld(),FakeBall(),*inputs(),action_model=correction_model())


def test_temporal_v8_migration_is_strict_and_adds_current_target_features():
    source = CausalRacketModeActionBallV3()
    checkpoint = {'schema': source.schema, 'model': source.state_dict()}
    restored = migrate_temporal_v8(checkpoint)
    assert type(restored) is TemporalCausalRacketModeTargetActionBallV8
    assert restored.input_scale.shape == (84,)
    assert restored.head.net[0].weight.shape[1] == 84
    assert torch.equal(restored.head.net[0].weight[:, :48],
                       source.head.net[0].weight)
    assert torch.equal(restored.head.net[6].weight, source.head.net[6].weight)


def test_temporal_v8_requires_current_target_and_preserves_command_gradient():
    world, ball = FakeWorld(), ModeBall(torch.tensor(.8))
    model = TemporalCausalRacketModeTargetActionBallV8(preview=FakePreview())
    with torch.no_grad():
        model.head.net[-1].bias[3] = 1.
    state, target, prev, feet, present, elapsed, launch = inputs()
    target = target.requires_grad_(True)
    predicted, _, _ = world(state, target, prev, feet, present, elapsed, launch)
    result = ball_transition(world, ball, state, target, prev, feet, present,
                             elapsed, launch, action_model=model)[0]
    grad = torch.autograd.grad(result[:, 52].sum(), target)[0]
    assert torch.isfinite(grad).all()
    assert grad[:, 12].abs().max() > 1e-7


def test_temporal_v9_migration_keeps_explicit_floor_schema():
    source = CausalRacketModeActionBallV3()
    restored = migrate_temporal_v9({'schema': source.schema, 'model': source.state_dict()})
    assert type(restored) is TemporalCausalRacketModeTargetActionBallV9
    assert restored.schema != TemporalCausalRacketModeTargetActionBallV8.schema
    assert restored.support_floor > 0 and restored.mode_floor > 0


def test_temporal_v10_migration_has_zero_initialized_linear_channel():
    source = CausalRacketModeActionBallV3()
    restored = migrate_temporal_v10({'schema': source.schema, 'model': source.state_dict()})
    assert type(restored) is TemporalCausalRacketModeTargetLinearActionBallV10
    assert torch.equal(restored.command_head.weight,
                       torch.zeros_like(restored.command_head.weight))


def test_temporal_v11_migration_has_zero_initialized_native_response_head():
    source = CausalRacketModeActionBallV3()
    restored = migrate_temporal_v11({'schema': source.schema, 'model': source.state_dict()})
    assert type(restored) is TemporalCausalRacketModeNativeResponseBallV11
    assert torch.equal(restored.native_response_head.net[-1].weight,
                       torch.zeros_like(restored.native_response_head.net[-1].weight))
    assert restored.native_response_head.net[0].weight.shape[1] == 84


def test_temporal_v11_present_only_calibration_gate_does_not_depend_on_missed_impact_probability():
    world=FakeWorld();args=inputs()
    model=TemporalCausalRacketModeNativeResponseBallV11(
        preview=FakePreview(), native_response_gate_mode='present_only')
    with torch.no_grad():
        model.native_response_head.net[-1].bias[3]=1.
    _,_,_,details_free=ball_transition(
        world,ModeBall(torch.tensor(0.)),*args,action_model=model,return_details=True)
    _,_,_,details_contact=ball_transition(
        world,ModeBall(torch.tensor(1.)),*args,action_model=model,return_details=True)
    torch.testing.assert_close(
        details_free['action_model_info']['native_response'],
        details_contact['action_model_info']['native_response'])
    assert details_free['action_model_info']['native_response_gate_mode']=='present_only'


def test_temporal_v12_migration_adds_explicit_command_delta_channel():
    source = CausalRacketModeActionBallV3()
    restored = migrate_temporal_v12({'schema': source.schema, 'model': source.state_dict()})
    assert type(restored) is TemporalCausalRacketModeNativeResponseBallV12
    assert restored.native_response_head.net[0].weight.shape[1] == 102
    assert restored.native_input_scale.shape == (102,)
    assert torch.equal(restored.native_response_head.net[-1].weight,
                       torch.zeros_like(restored.native_response_head.net[-1].weight))


def test_temporal_v12_explicit_command_delta_is_causal_and_differentiable():
    world, ball = FakeWorld(), ModeBall(torch.tensor(.8))
    model = TemporalCausalRacketModeNativeResponseBallV12(
        preview=FakePreview(), native_response_gate_mode='present_only')
    with torch.no_grad():
        model.native_response_head.net[-1].weight[3, -18] = 1.
    state, target, prev, feet, present, elapsed, launch = inputs()
    target = target.requires_grad_(True)
    result = ball_transition(world, ball, state, target, prev, feet, present,
                             elapsed, launch, action_model=model)[0]
    grad = torch.autograd.grad(result[:, 52].sum(), target)[0]
    assert torch.isfinite(grad).all()
    assert grad[:, 12].abs().max() > 1e-7


def test_temporal_v13_migration_adds_zero_initialized_toi_impulse_head():
    source = CausalRacketModeActionBallV3()
    restored = migrate_temporal_v13({'schema': source.schema, 'model': source.state_dict()})
    assert type(restored) is TemporalCausalRacketModeImpulseResidualBallV13
    assert restored.impulse_residual_head.net[0].weight.shape[1] == 84
    assert torch.equal(restored.impulse_residual_head.net[-1].weight,
                       torch.zeros_like(restored.impulse_residual_head.net[-1].weight))


def test_temporal_v13_applies_residual_at_toi_and_propagates_target_gradient():
    class Prior:
        def __call__(self, state, geometry, present, predicted_robot=None):
            zeros = state[:, 49:].clone()
            return zeros, dict(
                active=present.bool(),
                normal=state.new_tensor([[0., 0., 1.]]).expand(len(state), -1),
                pre_position=state[:, 49:52],
                pre_velocity=state[:, 52:55],
                impulse_dv=torch.zeros_like(state[:, 52:55]),
                remaining_time=state.new_full((len(state),), .01),
            )

        @staticmethod
        def advance(position, velocity, duration):
            return position + duration[:, None] * velocity, velocity

    world, ball = FakeWorld(), ModeBall(torch.tensor(.8))
    model = TemporalCausalRacketModeImpulseResidualBallV13(preview=FakePreview())
    model.__dict__['impulse_prior'] = Prior()
    with torch.no_grad():
        model.impulse_residual_head.net[-1].bias[0] = 1.
    state, target, prev, feet, present, elapsed, launch = inputs()
    target = target.requires_grad_(True)
    predicted, _, _, details = ball_transition(
        world, ball, state, target, prev, feet, present, elapsed, launch,
        action_model=model, return_details=True)
    assert details['impulse_action_residual'] is not None
    assert details['impulse_action_residual'][:, 5].abs().max() > 0
    grad = torch.autograd.grad(predicted[:, 52].sum(), target)[0]
    assert torch.isfinite(grad).all()
