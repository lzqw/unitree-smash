from types import SimpleNamespace

import torch

from quadsmash.rl.supermimic_physical_loop_replay_v1 import _stage_ids_from_tensors, _stage_window_masks
from quadsmash.rl.supermimic_recovery_batches import late_recovery_batch
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1
from quadsmash.rl.supermimic_toi_racket_objective import first_racket_cost, string_boundary


def test_lost_support_and_recontact_after_landing_are_distinct_from_first_touchdown():
    fields = [SimpleNamespace(path='auxiliary/contact_recovery/qualified_landing', width=1)]
    d = dict(event_state=torch.zeros(1, 4, 8), next_event_state=torch.zeros(1, 4, 8),
             feet=torch.zeros(1, 4, 4), next_feet=torch.zeros(1, 4, 4),
             elapsed=torch.full((1, 4), 4.), complete_reward=torch.zeros(1, 4),
             values=torch.tensor([[[0.], [0.], [1.], [1.]]]), valid=torch.ones(1, 4, 1))
    d['event_state'][..., 0] = d['next_event_state'][..., 0] = 1
    d['next_feet'][0, [1, 3], 2] = 1
    stages = _stage_ids_from_tensors(d, SimpleNamespace(fields=fields))
    masks = _stage_window_masks(d, torch.ones(1, 4, dtype=torch.bool), 1, stages)
    assert stages.tolist() == [[2, 4, 7, 7]]
    assert masks['touchdown'].tolist() == [[False, True, False, False]]
    assert masks['post_landing_support_loss'].tolist() == [[False, False, True, True]]


def test_risk_quota_and_fill_use_unique_episode_ids():
    n = 32
    before = dict(episode_index=torch.zeros(n), elapsed_tick=torch.full((n,), 500),
                  physical=dict(elapsed_s=torch.full((n,),5.), foot_contact=torch.zeros(n,4),
                                root_quaternion_w=torch.tensor([[1.,0,0,0]]).repeat(n,1)),
                  auxiliary=dict(jump=dict(landing_time=torch.ones(n)),
                                 contact_recovery=dict(landing_time=torch.full((n,),3.))))
    pools = {}
    for i, name in enumerate(('pre_body_ground','touchdown','stable_recovery')):
        mask = torch.zeros(n,dtype=torch.bool)
        mask[8*i:8*i+8] = True
        pools[name] = [(dict(before=before, _failure_lead_ticks=40), mask)]
    _, logs=late_recovery_batch(pools,16,1)
    assert [len(r['samples']) for r in logs] == [8,4,4]
    assert len({v['lane'] for r in logs for v in r['samples']}) == 16
    pools.pop('pre_body_ground')
    _, logs=late_recovery_batch(pools,16,1)
    assert len({v['lane'] for r in logs for v in r['samples']}) == 16


def test_first_racket_compares_three_dimensional_parts_with_body_only_gradient(monkeypatch):
    import quadsmash.rl.supermimic_toi_racket_objective as toi
    from quadsmash.physics.full_racket_v4 import load_racket, string_strips
    path='configs/full_racket_v4_r3.json'
    like = torch.zeros(2, 55, requires_grad=True)
    def racket(body, geom):
        n=len(body)
        return torch.cat((body[:,:3], torch.eye(3).reshape(1,9).expand(n,9),body[:,7:13]),dim=-1)
    monkeypatch.setattr(toi, 'racket_state', racket)
    preview = BallGeometryPreviewV1()
    strips = len(string_strips(load_racket(path)))
    local = torch.tensor([[.026,0.,0.],[.026,.111,.010]],requires_grad=True)
    impact = dict(time=torch.full((2,),.006,requires_grad=True),pre_position=local,local=local)
    cost = first_racket_cost(like, like, impact, None, preview, strips, string_boundary(path, like), .01)
    assert cost[1] > cost[0]
    grads=torch.autograd.grad(cost.sum(),(like,local,impact['time']),allow_unused=True)
    assert grads[0][1].norm()>0 and all(g is None for g in grads[1:])


def test_native_support_loss_samples_the_transition_even_before_any_unloaded_state():
    from types import SimpleNamespace
    from quadsmash.rl.supermimic_physical_loop_replay_v1 import post_landing_unload_windows
    paths = ['auxiliary/contact_recovery/qualified_landing', 'rally/released']
    schema = SimpleNamespace(fields=[SimpleNamespace(path=p,width=1) for p in paths])
    d = dict(feet=torch.ones(3,6,4), next_feet=torch.ones(3,6,4),
             values=torch.ones(3,6,2), valid=torch.ones(3,6,2), elapsed=torch.ones(3,6),
             event_state=torch.zeros(3,6,8))
    d['next_feet'][1] = 0
    d['feet'][1,1] = 0
    d['values'][1,2,0] = 0
    d['values'][1,3,1] = 0
    d['event_state'][1,4,5] = 1
    valid = torch.ones(2,6,dtype=torch.bool)
    valid[0,5] = False
    actual = post_landing_unload_windows(d,schema,2,valid)
    assert actual[0].tolist() == [True,False,False,False,False,False]
    assert actual[1].tolist() == [True,False,False,False,False,True]
