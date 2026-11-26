import torch
from scripts.native_recovery_ppo import gae


def test_gae_uses_critic_and_respects_gate_and_episode_boundary():
    r=torch.tensor([[2.],[3.],[4.],[999.]])
    v=torch.tensor([[1.],[1.],[2.],[0.]])
    a=torch.tensor([[False],[True],[True],[False]])
    d=torch.tensor([[False],[False],[True],[False]])
    out=gae(r,v,a,d,gamma=.5,lam=.5)
    torch.testing.assert_close(out,torch.tensor([[0.],[3.5],[2.],[0.]]))
