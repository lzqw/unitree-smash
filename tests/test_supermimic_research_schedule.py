from types import SimpleNamespace

import torch

from quadsmash.rl.supermimic_alternating_schedule_v1 import make_plan
from quadsmash.rl.supermimic_outgoing_flight import outgoing_cost
from quadsmash.rl.supermimic_training_stage_v1 import resolve_training_stage


def test_256_retained_and_two_combined_refreshes():
    events=make_plan(128,128,256,256,refresh_every_pairs=8)
    blocks=[e for e in events if e['kind']=='train']
    assert [e['head'] for e in blocks]==['recovery','approach']*16
    assert sum(e['target'] for e in blocks)==256
    assert sum(e['proposal_limit'] for e in blocks)==512
    assert [e['pair'] for e in events if e['kind']=='refresh']==[7,15]
    stage=resolve_training_stage('research_contact')
    assert stage.contact_bptt_mode=='enabled' and not stage.promotable


def test_outgoing_cost_weak_return_has_forward_descent_direction():
    prior=SimpleNamespace(gravity=torch.tensor([0.,0.,-9.81]),wind=torch.zeros(3),drag=.02)
    court=dict(net_plane_x_m=1.5,target_x_limits_m=[2.,4.],target_y_half_width_m=1.,
               net_side_top_m=1.55,net_center_top_m=1.524,post_y_center_abs_m=3.075)
    ball=torch.tensor([[0.,0.,2.,1.,0.,1.]],requires_grad=True)
    cost=outgoing_cost(ball,prior,court)
    g,=torch.autograd.grad(cost.sum(),ball)
    plus=ball.detach().clone()
    plus[:,3]+=.001
    torch.testing.assert_close(g[:,3],(outgoing_cost(plus,prior,court)-cost.detach())/.001,atol=.002,rtol=.003)
    assert g[0,3]<0
