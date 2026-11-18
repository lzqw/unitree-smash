"""Interpolation and gradient contracts; no native accuracy claim."""
import torch
from torch import nn
from quadsmash.rl.supermimic_impact_timing_v1 import (
    hermite_position_velocity,interpolate_physical_state,causal_sweep_time)
from quadsmash.rl.supermimic_action_ball_v1 import ImpactTimeActionBallV3,action_ball_for_schema
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from test_supermimic_action_ball_v1 import FakeWorld,inputs
from test_supermimic_racket_mode_action import ModeBall


class PlanePreview(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('times',torch.linspace(0.,.01,161))
    def racket_gap(self,points):return points[...,0]-.025


def test_constant_acceleration_has_quadratic_early_command_position_effect():
    acceleration=torch.tensor([[2.]],requires_grad=True)
    zero=torch.zeros_like(acceleration);dt=.01;time=torch.tensor([.001])
    end=.5*acceleration*dt*dt;velocity=acceleration*dt
    p,v=hermite_position_velocity(zero,zero,end,velocity,time,dt)
    assert torch.allclose(p,.5*acceleration*time[:,None].square(),atol=1e-10)
    assert torch.allclose(v,acceleration*time[:,None],atol=1e-9)
    grad=torch.autograd.grad(p.sum(),acceleration)[0]
    assert torch.allclose(grad,torch.full_like(grad,.5e-6),atol=1e-10)


def test_impact_root_uses_causal_motion_and_keeps_within_bin_gradient():
    first=torch.tensor([[.1,0.,0.]]);velocity=torch.tensor([[-10.,0.,0.]])
    last=torch.tensor([[.0002,0.,0.]],requires_grad=True);preview=PlanePreview()
    time,info=causal_sweep_time(first,velocity,last,preview)
    assert info['crossing_found'].tolist()==[True]
    grad=torch.autograd.grad(time.sum(),last)[0]
    epsilon=1e-5;direction=torch.tensor([[epsilon,0.,0.]])
    plus=causal_sweep_time(first,velocity,last.detach()+direction,preview)[0]
    minus=causal_sweep_time(first,velocity,last.detach()-direction,preview)[0]
    assert torch.allclose(grad[:,0],(plus-minus)/(2*epsilon),atol=5e-5,rtol=1e-3)
    assert time.item()>.0075 and time.item()<.0076


def test_no_crossing_is_explicit_not_certified_contact():
    first=torch.tensor([[1.,0.,0.]]);velocity=torch.zeros_like(first)
    time,info=causal_sweep_time(first,velocity,first,PlanePreview())
    assert not info['crossing_found'].any()
    assert torch.equal(time,torch.zeros_like(time))


def test_physical_interpolation_matches_endpoints_and_quaternion_sign():
    state=torch.zeros(2,55);state[:,3]=1.;end=state.clone()
    end[:,0]=.1;end[:,13]=.02;end[:,3]=-1.
    before=state.clone();after=end.clone()
    out=interpolate_physical_state(state,end,torch.tensor([0.,.01]),.01)
    assert torch.equal(out[0],state[0])
    assert torch.equal(out[1,:3],end[1,:3])
    assert torch.equal(out[1,7:],end[1,7:])
    assert torch.equal(out[:,3:7],state[:,3:7])
    assert torch.equal(state,before) and torch.equal(end,after)


def test_complete_ball_path_uses_impact_schema_with_finite_command_gradient():
    torch.manual_seed(11)
    model=action_ball_for_schema(ImpactTimeActionBallV3.schema,preview=PlanePreview())
    with torch.no_grad():model.head.net[-1].weight.fill_(.01)
    state,target,*other=inputs();state[:,49]=.06;target.requires_grad_(True)
    prediction=ball_transition(FakeWorld(),ModeBall(torch.tensor(.8)),state,target,*other,action_model=model)[0]
    grad=torch.autograd.grad(prediction[:,52:].sum(),target)[0]
    assert torch.isfinite(grad).all() and grad.abs().max()>1e-7
