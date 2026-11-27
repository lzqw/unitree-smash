"""Numerical phase gate against the independent Pinocchio dynamics algorithms."""
from pathlib import Path
import numpy as np
import pinocchio as pin
import torch

from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.rl.dynamimic_lagrangian_v1 import DynaMimicLagrangianV1

ASSET = Path('/home/autodrive/LZQW/unitree_smash/assets/quadsmash_full_racket_v4_r3.usd')


def test_landing_dynamics_512_configurations():
    ref = LoadedPinocchioModelV1()
    module = DynaMimicLagrangianV1(ASSET)
    rng = np.random.default_rng(322)
    q = rng.normal(size=(512,25))
    q[:,2] = rng.uniform(.15,.6,512)
    q[:,3:7] /= np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
    v = rng.normal(size=(512,24))
    q_t = torch.from_numpy(q).requires_grad_()
    mass, bias, feet, speed, jac = module(q_t,torch.from_numpy(v))
    order = np.r_[np.arange(6),ref.v_indices]
    errors = np.zeros(5)
    for i in range(len(q)):
        cfg = pin.neutral(ref.model)
        cfg[:3],cfg[3:7],cfg[ref.q_indices] = q[i,:3],q[i,[4,5,6,3]],q[i,7:]
        vel = np.zeros(24)
        vel[order] = v[i]
        snap = ref.snapshot(cfg,vel)
        j = snap['foot_jacobian'][:,:,order]
        expected = (snap['mass_matrix'][np.ix_(order,order)],snap['nonlinear'][order],
                    snap['body_positions_world'][[ref.names.index(f) for f in ref.feet]],j @ v[i],j)
        errors = np.maximum(errors,[np.abs(t[i].detach().numpy()-e).max()
                                    for t,e in zip((mass,bias,feet,speed,jac),expected)])
    print('Maximum errors (M, h, foot position, velocity, J):',errors.tolist())
    assert errors.max() < 1e-5
    gradients = torch.autograd.grad(mass.square().mean()+bias.square().mean()+jac.square().mean(),q_t)[0]
    assert gradients.isfinite().all() and gradients[:,7:].abs().sum() > 0
