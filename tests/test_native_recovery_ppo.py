import torch
from scripts.native_recovery_ppo import gae


def test_gae_uses_critic_and_respects_gate_and_episode_boundary():
    r=torch.tensor([[2.],[3.],[4.],[999.]])
    v=torch.tensor([[1.],[1.],[2.],[0.]])
    a=torch.tensor([[False],[True],[True],[False]])
    d=torch.tensor([[False],[False],[True],[False]])
    out=gae(r,v,a,d,gamma=.5,lam=.5)
    torch.testing.assert_close(out,torch.tensor([[0.],[3.5],[2.],[0.]]))


def test_first_body_contact_keeps_terminal_penalty_and_excludes_later_rewards():
    rewards = torch.tensor([[0.], [-15.], [50.], [100.]])
    active = torch.tensor([[True], [True], [False], [False]])
    terminal = torch.tensor([[False], [True], [True], [True]])
    result = gae(rewards, torch.zeros_like(rewards), active, terminal, gamma=.5, lam=1.)
    torch.testing.assert_close(result, torch.tensor([[-7.5], [-15.], [0.], [0.]]))


def test_two_phase_reward_and_native_longest_window():
    from types import SimpleNamespace
    from scripts.native_recovery_ppo import reward_state

    n = 4
    zeros = torch.zeros(n)
    g = torch.tensor([[0., 0., -1.], [0., .8, -.6]]).repeat(2, 1)
    data = SimpleNamespace(projected_gravity_b=g, root_link_lin_vel_w=torch.zeros(n,3),
                           root_ang_vel_w=torch.zeros(n,3), joint_vel=torch.zeros(n,18),
                           root_pos_w=torch.tensor([[0.,0.,.34]]).repeat(n,1),
                           body_link_pos_w=torch.zeros(n,1,3))
    r = SimpleNamespace(stable_run=zeros, longest_stable=zeros+.7,
                        relaxed_success=zeros.bool(), stable_seen=zeros.bool())
    env = SimpleNamespace(robot=SimpleNamespace(data=data), scene=SimpleNamespace(env_origins=torch.zeros(n,3)),
                          cfg=SimpleNamespace(net_x_m=1.5), racket_body_id=0, leg_ids=list(range(12)),
                          observed_foot_contact=torch.ones(n,4), landed_once=torch.tensor([0,0,1,1]).bool(),
                          contact_recovery=r, body_ground_seen=zeros.bool(),
                          rally=SimpleNamespace(robot_net_fault=zeros.bool()))
    dense, strict_tick, longest, *_ = reward_state(env)
    assert torch.all(dense >= 0)
    assert dense[0] > dense[1] and dense[2] > dense[3]
    assert not strict_tick.any()
    torch.testing.assert_close(longest, zeros+.7)


def test_targets_frozen_even_when_value_function_changes():
    from scripts.native_recovery_ppo import rollout_targets
    v = torch.tensor([[1.], [2.]], requires_grad=True)
    reward = torch.tensor([[0.], [3.]])
    active = torch.ones_like(reward, dtype=torch.bool)
    done = torch.tensor([[False], [True]])
    adv, ret = rollout_targets(reward, v, active, done, False)
    expected = ret.clone()
    with torch.no_grad():
        v.fill_(100.)
    assert not adv.requires_grad and not ret.requires_grad
    torch.testing.assert_close(ret, expected)
    torch.testing.assert_close(ret[-1], torch.tensor(3.))


def test_halving_preserves_previous_iterations_and_heldout_guard():
    from scripts.native_recovery_ppo import halve_update, heldout_regression
    head = torch.nn.Linear(2, 1)
    start = {k: p.detach().clone() for k,p in head.named_parameters()}
    with torch.no_grad():
        for p in head.parameters():
            p.add_(2.)
    halve_update(head, start)
    for k,p in head.named_parameters():
        torch.testing.assert_close(p, start[k]+1.)
    base = dict(relaxed=8, body=20)
    assert not heldout_regression(base, base)
    assert not heldout_regression(base, dict(relaxed=7, body=20))
    assert heldout_regression(base, dict(relaxed=6, body=20))
    assert not heldout_regression(base, dict(relaxed=8, body=21))
    assert heldout_regression(base, dict(relaxed=8, body=22))


def test_lift_has_low_height_slope_and_relaxes_leg_damping():
    from scripts.native_recovery_ppo import lift_and_damping
    height = torch.tensor([.14, .15, .23, .24])
    slow = lift_and_damping(height, torch.zeros(4), torch.full((4,), .08))
    fast = lift_and_damping(height, torch.full((4,), 4.), torch.full((4,), .08))
    torch.testing.assert_close(slow[1]-slow[0], torch.tensor(.06/18))
    torch.testing.assert_close(slow[0]-fast[0], .5*(slow[2]-fast[2]))


def test_low_root_terminal_filters_timeout_success_and_existing_faults():
    from scripts.native_recovery_ppo import low_root_terminal
    eye = torch.eye(6, dtype=torch.bool)
    result = low_root_terminal(~eye[0],eye[1],eye[2],eye[3],eye[4],torch.full((6,),.11))
    torch.testing.assert_close(result, eye[5])


def test_macro_likelihood_scores_actuator_stops_by_probability_mass():
    from scripts.native_recovery_ppo import exploration_center, executed_log_prob
    mean = torch.tensor([[-1.22, 0., 1.22]], requires_grad=True)
    center = exploration_center(mean)
    action = torch.tensor([[-1., .05, 1.]])
    sigma = torch.tensor(.1)
    normal = torch.distributions.Normal(center, sigma)
    expected = normal.cdf(action)[0,0].log() + normal.log_prob(action)[0,1] + (1-normal.cdf(action)[0,2]).log()
    result = executed_log_prob(action, center, sigma)
    torch.testing.assert_close(result.squeeze(), expected)
    result.backward()
    assert (mean.grad.abs() > 0).all()


def test_macro_partial_terminal_keeps_last_reward_only_once():
    from scripts.native_recovery_ppo import decision_targets
    rewards = torch.ones(7, 1)
    values = torch.zeros_like(rewards)
    live = torch.tensor([True]*6+[False])[:,None]
    done = torch.tensor([False]*5+[True,False])[:,None]
    decisions = torch.tensor([True,False,False,False,True,False,False])[:,None]
    adv, targets = decision_targets(rewards,values,live,done,decisions,False,gamma=1.)
    torch.testing.assert_close(targets,torch.tensor([5.9,2.]))
    torch.testing.assert_close(adv, targets)


def test_native_thigh_clearance_uses_rotated_authored_box():
    from types import SimpleNamespace
    import json
    import itertools
    from quadsmash.rl.dynamimic_ground_clearance import SHAPES
    from scripts.native_recovery_ppo import NativeThighClearance
    geom = json.loads(SHAPES.read_text())
    shape, = [s for s in geom['shapes'] if s['body']=='RL_thigh']
    data = SimpleNamespace(body_link_quat_w=torch.tensor([[[.5,.5,.5,.5]]]),
                           body_link_pos_w=torch.tensor([[[0.,0.,1.3]]]))
    env = SimpleNamespace(robot=SimpleNamespace(body_names=['RL_thigh'],data=data), device='cpu',
                          scene=SimpleNamespace(env_origins=torch.tensor([[0.,0.,1.]])))
    geometry = NativeThighClearance(env, next(iter(geom['asset_sha256'].values())))
    vertices = torch.tensor(list(itertools.product([-.5,.5],repeat=3))) * shape['size']
    transform = torch.tensor(shape['transform'])
    body_vertices = vertices @ transform[:3,:3] + transform[3,:3]
    # A 120-degree rotation about (1,1,1) sends local Y into world Z.
    torch.testing.assert_close(geometry(env), (.3 + body_vertices[:,1].min()).reshape(1))
