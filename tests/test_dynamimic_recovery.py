"""Numerical phase gate against the independent Pinocchio dynamics algorithms."""
from pathlib import Path
import numpy as np
import pinocchio as pin
import pytest
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
    data = dict(body=body,root_terminated=torch.zeros_like(body),active=live & ~body,obs=obs,live=live,release_time=torch.zeros(t,n),
                landed=landed,strict=torch.zeros_like(live))
    data['root_terminated'][122,1] = True
    groups = recovery_start_groups(data,[2,8,16,48,96],torch.tensor([False,False,True]))
    assert [8,0] in groups[0].tolist() and [2,0] not in groups[0].tolist()
    assert [48,0] in groups[1].tolist() and [48,1] in groups[2].tolist()
    assert [96,1] in groups[1].tolist() and [96,1] not in groups[2].tolist()
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

    for available, expected in ((0, 0), (5, 5), (60, 12), (200, 51)):
        urgent = torch.zeros(3, 200, dtype=torch.bool)
        urgent[1, :available] = True
        rows = stratified_starts(groups, 128, torch.Generator().manual_seed(17), imminent=urgent)
        assert rows[:, 0].bincount().tolist() == [38, 51, 39]
        assert int((rows[:, 1] % 2).sum()) == 64
        assert len(rows.unique(dim=0)) == 128
        assert int(urgent[rows[:, 0], rows[:, 1]].sum()) == expected


def test_native_critic_retains_return_units_after_reload_and_reads_old_fits():
    from scripts.fit_dynamimic_recovery_critic import NativeTailValue
    value = NativeTailValue(torch.nn.Identity())
    with torch.no_grad():
        value.value[-1].weight.fill_(.01)
    legacy = {'critic': value.value.state_dict()}
    value.load_native_fit(legacy)
    obs = torch.randn(8, 794, requires_grad=True)
    initial = value(obs)
    value.load_native_fit(dict(legacy, return_scale=17., return_mean=-3.))
    scaled = value(obs)
    torch.testing.assert_close(scaled, 17 * initial - 3.)
    assert torch.autograd.grad(scaled.sum(), obs)[0].norm() > 0
    value.load_native_fit(legacy)
    torch.testing.assert_close(value(obs), initial)


def test_candidate_steps_remain_distinct_after_budget_halvings():
    from scripts.train_dynamimic_recovery_bptt import candidate_steps
    moderate, conservative = candidate_steps(lambda alpha: .076 * alpha)
    assert moderate == ('moderate', .035, .25, .019)
    assert conservative == ('conservative', .020, .125, .0095)


def test_root_failure_charged_once_and_blocks_tail_after_rebound():
    from scripts.train_dynamimic_recovery_bptt import root_survival
    from scripts.fit_dynamimic_recovery_critic import root_termination
    height = torch.tensor([.23, .12, .10], requires_grad=True)
    s = root_survival(torch.ones_like(height), height)
    torch.testing.assert_close(s, torch.tensor([1., .5, 0.]))
    torch.testing.assert_close(root_survival(s, torch.full_like(s,.34)), s)
    s.sum().backward()
    assert height.grad[1] > 0
    assert root_termination(torch.tensor([True,False,True]), height).tolist() == [False,False,True]


def test_first_block_diagnostic_scores_rejected_candidate_without_promoting():
    from scripts.run_dynamimic_recovery_loop import choose_candidate, first_block_candidate_to_score
    baseline = dict(qualified=54, body=32, strict=3, relaxed=8, ball=46, net=0)
    trials = [dict(baseline, ball=45, body=30), dict(baseline, ball=45, body=28)]
    selection = choose_candidate(baseline, trials)
    assert selection is None
    assert first_block_candidate_to_score(selection, trials) == 1
    assert first_block_candidate_to_score(0, trials) == 0


def test_retention_ridge_balances_case_phases_and_excludes_validation():
    from scripts.fit_dynamimic_recovery_retention import ridge_correction

    x = torch.tensor([[1., 1.], [2., 1.], [3., 1.], [4., 1.]], dtype=torch.double)
    y = x[:, :1].clone()
    success = torch.tensor([True, True, False, True])
    keys = torch.arange(4)
    training = torch.tensor([True, True, True, False])
    expected = ridge_correction(x, y, success, keys, training)
    y[-1] = 10000
    ix = torch.tensor([0, 0, 0, 1, 2, 3])
    actual = ridge_correction(x[ix], y[ix], success[ix], keys[ix], training[ix])
    torch.testing.assert_close(actual, expected)


def test_native_body_fault_does_not_fit_held_lanes_and_remains_differentiable():
    from scripts.dynamimic_body_fault import body_fault_score, fit_native_body_fault
    from quadsmash.rl.dynamimic_recovery_wm_v1 import local_robot_state

    rng = torch.Generator().manual_seed(7)
    state = torch.randn((24, 16, 55), generator=rng)
    active = torch.ones(24, 16, dtype=torch.bool)
    fault = torch.zeros_like(active)
    fault[19, 2:8] = True
    data = dict(after=state, feet=torch.rand(24, 16, 4, generator=rng), active=active,
                body=fault, strict=torch.zeros_like(active), relaxed=torch.zeros_like(active))
    before, _ = fit_native_body_fault(data)
    data['body'][10, :2] = True
    after, metrics = fit_native_body_fault(data)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert metrics['first_body']['cases'] == 2
    q, v = local_robot_state(state[4])
    q.requires_grad_()
    score = body_fault_score(q, v, data['feet'][4], after)
    grad, = torch.autograd.grad(score.sum(), q)
    assert torch.isfinite(grad).all() and grad.abs().max() > 0


def test_fixed128_rechecks_native_ball_and_recovery_contract():
    from scripts.run_dynamimic_recovery_loop import passes_fixed128

    score = dict(qualified_contacts=108, relaxed_ball_side=95, robot_net=0,
                 body_ground_given_contact=63, strict_physical_recovery=16, relaxed_physical_recovery=35)
    assert passes_fixed128(score)
    for key, failing in [('qualified_contacts', 107), ('relaxed_ball_side', 94), ('robot_net', 1),
                         ('body_ground_given_contact', 64), ('strict_physical_recovery', 15),
                         ('relaxed_physical_recovery', 34)]:
        assert not passes_fixed128(dict(score, **{key: failing}))


def test_native_replay_keeps_case_and_launch_splits_across_training_panels():
    from scripts.fit_dynamimic_recovery_wm import verify_replay_cases

    def panel(offset):
        return [dict(case_id=str(i+offset), launch=[i+offset]) for i in range(256)]

    first, second = panel(0), panel(256)
    verify_replay_cases([first, second, first])
    second[4] = dict(case_id='renamed', launch=first[0]['launch'])
    with pytest.raises(ValueError, match='partition'):
        verify_replay_cases([first, second])
    second[4] = dict(case_id=first[4]['case_id'], launch=[9999])
    with pytest.raises(ValueError, match='case'):
        verify_replay_cases([first, second])
