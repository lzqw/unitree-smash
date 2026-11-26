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
    assert dense[0] > dense[1] and dense[2] > dense[3]
    assert not strict_tick.any()
    torch.testing.assert_close(longest, zeros+.7)
