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
                                   effort_limit=[23.5]*12+[100.]*6, velocity_limit=[30.]*12+[5.]*6,
                                   position_limits=[[-3., 3.]]*18)])
    wm = DynaMimicRecoveryWorldModel(ASSET, contract).double()
    state = torch.zeros(2, 55, dtype=torch.float64)
    state[:, 2] = .38
    state[:, 3] = 1.
    q, v = local_robot_state(state)
    target = torch.full_like(q[:, 7:], .01, requires_grad=True)
    sequential = []
    for member in range(3):
        q1, v1, aux = wm(q, v, target, member)
        sequential.append((q1.detach(), v1.detach(), aux['foot_contact'].detach()))
        force = aux['contact_force']
        assert (force[..., 2] >= 0).all()
        assert (force[..., :2].norm(dim=-1) <= .9*force[..., 2]+1e-12).all()
        assert torch.allclose(q1[:, 3:7].norm(dim=-1), torch.ones_like(q1[:, 0]))
        grad, = torch.autograd.grad(v1.square().mean(), target)
        assert grad.isfinite().all() and grad.abs().sum() > 0

    with torch.no_grad():
        qb, vb, aux = wm(q.repeat(3, 1), v.repeat(3, 1), target.repeat(3, 1), (0, 1, 2))
        for actual, parts in zip((qb, vb, aux['foot_contact']), zip(*sequential)):
            torch.testing.assert_close(actual, torch.cat(parts))


def test_recovery_starts_require_release_and_keep_prefall_states():
    from scripts.train_dynamimic_recovery_bptt import recovery_start_groups
    n, t = 3, 130
    live = torch.ones(t,n,dtype=torch.bool)
    landed = live.clone()
    landed[:15] = False
    body = torch.zeros_like(live)
    body[80:,0] = True
    obs = torch.zeros(t,n,794)
    obs[:,:,775] = torch.arange(t)[:,None]*.01
    data = dict(body=body,active=live & ~body,obs=obs,live=live,release_time=torch.zeros(t,n),
                landed=landed,strict=torch.zeros_like(live))
    groups = recovery_start_groups(data,[2,8,16,48,96],torch.tensor([False,False,True]))
    assert [8,0] in groups[0].tolist() and [2,0] not in groups[0].tolist()
    assert [48,0] in groups[1].tolist() and [48,1] in groups[2].tolist()
    assert all(not (pool[:,1]==2).any() for pool in groups)


def test_native_promotion_keeps_ball_side_and_rejects_robot_net():
    from scripts.run_dynamimic_recovery_loop import choose_candidate
    old = dict(qualified=50,body=28,net=0,ball=39,relaxed=12,strict=6)
    assert choose_candidate(old,[old,dict(old,strict=7)]) == 1
    assert choose_candidate(old,[dict(old,body=26,net=1),dict(old,strict=7,ball=38)]) is None


def test_recovery_only_saturation_gradient_and_focus_coverage():
    from scripts.train_dynamimic_recovery_bptt import saturation_cost, focus_cost
    raw = torch.tensor([1.1,-1.2,.5],requires_grad=True)
    saturation_cost(raw).backward()
    assert raw.grad[0]>0 and raw.grad[1]<0 and raw.grad[2]==0
    w = torch.ones(128,794,requires_grad=True)
    focus_cost(w,torch.zeros_like(w)).backward()
    assert (w.grad[:,:759].reshape(128,11,69)[:,:,61:67]>0).all()
    assert (w.grad[:,776:794]>0).all()
    assert (w.grad[:,:759].reshape(128,11,69)[:,:,:61]==0).all()


def test_head_sphere_is_a_separate_support_even_without_foot_contact():
    from quadsmash.rl.dynamimic_recovery_wm_v1 import solve_support_impulses

    # A body descending onto the fifth sphere with all four feet above ground.
    tensor = torch.float64
    mobility = torch.eye(24, dtype=tensor)[None]
    jac = torch.zeros(1, 5, 3, 24, dtype=tensor)
    jac[..., :3] = torch.eye(3, dtype=tensor)
    velocity = torch.zeros(1, 24, dtype=tensor)
    velocity[:, 2] = -2
    gap = torch.tensor([[.20]*4 + [0]], dtype=tensor)
    rates = torch.full((1, 18), 30., dtype=tensor)
    result, force = solve_support_impulses(mobility, jac, velocity, gap,
        torch.zeros(1, 5, 3, dtype=tensor), .0025, (-rates, rates))
    assert abs(result[0, 2]) < .04
    assert force[0, :4].abs().max() == 0
    assert force[0, 4, 2] > 750


def test_recovery_sampling_preserves_phase_mix_and_equal_approach_counts():
    from scripts.train_dynamimic_recovery_bptt import stratified_starts
    groups = [torch.stack((torch.full((200,), stage), torch.arange(200)), 1) for stage in range(3)]
    rows = stratified_starts(groups, 128, torch.Generator().manual_seed(17))
    assert rows[:, 0].bincount().tolist() == [38, 51, 39]
    assert int((rows[:, 1] % 2).sum()) == 64
    assert len(rows.unique(dim=0)) == 128
