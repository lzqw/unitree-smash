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


def test_contact_forces_and_integrator_propagate_target_gradients():
    from quadsmash.rl.dynamimic_recovery_wm_v1 import DynaMimicRecoveryWorldModel, local_robot_state
    from quadsmash.rl.supermimic_physical_observation_v1 import JOINT_NAMES
    import hashlib

    contract = dict(schema='native-joint-drive-v1', asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
                    control_dt_s=.01, joint_names=JOINT_NAMES,
                    profiles=[dict(kp=[25.]*12+[80.]*6, kd=[.5]*12+[4.]*6,
                                   effort_limit=[23.5]*12+[100.]*6)])
    wm = DynaMimicRecoveryWorldModel(ASSET, contract).double()
    state = torch.zeros(2, 55, dtype=torch.float64)
    state[:, 2] = .38
    state[:, 3] = 1.
    q, v = local_robot_state(state)
    target = torch.full_like(q[:, 7:], .01, requires_grad=True)
    for member in range(3):
        q1, v1, aux = wm(q, v, target, member)
        force = aux['contact_force']
        assert (force[..., 2] >= 0).all()
        assert (force[..., :2].norm(dim=-1) <= .9*force[..., 2]+1e-12).all()
        assert torch.allclose(q1[:, 3:7].norm(dim=-1), torch.ones_like(q1[:, 0]))
        grad, = torch.autograd.grad(v1.square().mean(), target)
        assert grad.isfinite().all() and grad.abs().sum() > 0
