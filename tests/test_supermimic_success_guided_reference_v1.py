"""Reference contracts using asset FK and synthetic labels, not native success evidence."""
import json
import pytest
import torch
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1, MODEL
from quadsmash.rl.supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1, task_space_features

def fixture():
    geometry=FullBodyGeometryV1(json.loads(MODEL.read_text())['asset_sha256'])
    t,n=12,3;state=torch.zeros(t,n,55);state[:,:,3]=1.;state[:,:,2]=.5
    state[:,:,49]=.7;state[:,:,51]=.5
    events=torch.zeros(t,n,8);events[2:,:,0]=1;events[5:,:2,1]=1;events[6:,:2,2:4]=1;events[10:,:2,4]=1
    events[-1,1,7]=1
    feet=torch.zeros(t,n,4,dtype=torch.bool);feet[:2]=True;feet[8:]=True
    done=torch.zeros(t,n,dtype=torch.bool);done[-1]=True
    data=dict(state=state,next_state=state.clone(),event_state=events,next_event_state=events.clone(),feet=feet,
              next_feet=torch.cat((feet[1:],feet[-1:])),episode=torch.zeros(t,n,dtype=torch.long),
              elapsed=torch.arange(t)[:,None].expand(t,n)*.01,terminated=torch.zeros_like(done),timeout=done)
    return geometry,data

def test_terminal_evidence_and_event_selection():
    g,d=fixture();bank=TaskSpaceReferenceBankV1.from_packed(d,g,source_sha256='fixture',policy_id='p0')
    assert bank.manifest['approach_count']==1 and bank.manifest['recovery_count']==1
    assert bank.entries[0]['anchor_time']==pytest.approx(.05)
    assert bank.entries[1]['anchor_time']==pytest.approx(.08)
    assert all(e['terminal_verified'] for e in bank.entries)
    del d['timeout']
    with pytest.raises(ValueError,match='missing'):TaskSpaceReferenceBankV1.from_packed(d,g)

def test_time_alignment_mask_and_finite_gradient():
    g,d=fixture();bank=TaskSpaceReferenceBankV1.from_packed(d,g)
    initial=d['state'][3,:1].clone();elapsed=torch.tensor([2.03]);event=torch.tensor([2.05])
    ref=bank.bind(initial,'approach',geometry=g,elapsed=elapsed,event_time=event)
    assert ref['matched'].all()
    predicted=initial.clone();predicted[:,49]+=.01;predicted.requires_grad_()
    loss=bank.loss(predicted,elapsed,ref,'approach',geometry=g,valid_mask=torch.tensor([True]))
    grad=torch.autograd.grad(loss,predicted)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    original=ref['matched'].clone()
    masked=bank.loss(predicted,elapsed,ref,'approach',geometry=g,valid_mask=torch.tensor([False]))
    assert float(masked)==0 and torch.equal(original,ref['matched'])
    # The racket normal matches the local X convention used by the ball model.
    feature=task_space_features(initial,g,'approach')
    assert feature.shape==(1,18)

def test_reject_distant_and_deduplicate():
    g,d=fixture();bank=TaskSpaceReferenceBankV1.from_packed(d,g,match_distance_threshold=.01)
    x=d['state'][3,:1].clone();x[:,49]+=100
    ref=bank.bind(x,'approach',geometry=g,elapsed=torch.tensor([.03]))
    assert not ref['matched'].any()
    count=len(bank.entries);bank.extend(bank);assert len(bank.entries)==count
    restored=TaskSpaceReferenceBankV1.from_payload(bank.payload()).to('cpu')
    assert restored.manifest==bank.manifest
    with pytest.raises(ValueError):TaskSpaceReferenceBankV1([],g.model_sha256,'',float('inf'))

def test_paid_event_latches_cannot_override_independent_ball_side_rejection():
    g,d=fixture()
    rows=[dict(env_id=i,episode_index=0,scoring_schema='historical-independent-complete5-v1',
               ball_side_complete_except_recovery=False,recovery_success=i==0,body_ground_seen=False) for i in range(3)]
    bank=TaskSpaceReferenceBankV1.from_packed(d,g,terminal_records=rows)
    assert bank.manifest['approach_count']==0 and bank.manifest['recovery_count']==1
    assert bank.manifest['protocol_verified']
    with pytest.raises(ValueError,match='Missing terminal score'):
        TaskSpaceReferenceBankV1.from_packed(d,g,terminal_records=rows[:1])


def test_holdout_never_enters_tracking_and_exclusion_survives_resume_and_extend():
    g,d=fixture()
    bank=TaskSpaceReferenceBankV1.from_packed(d,g,excluded_lanes=[0])
    assert not bank.entries
    restored=TaskSpaceReferenceBankV1.from_payload(bank.payload()).to('cpu')
    assert restored.manifest['excluded_lanes']==[0]
    leaking=TaskSpaceReferenceBankV1.from_packed(d,g)
    with pytest.raises(ValueError,match='heldout'):
        restored.extend(leaking)
    with pytest.raises(ValueError,match='heldout'):
        leaking.assert_excludes([0])
    clean=TaskSpaceReferenceBankV1.from_packed(d,g,excluded_lanes=[1,2])
    clean.assert_excludes([1,2])
    clean.extend(clean)
    assert clean.manifest['approach_count']==1 and clean.manifest['recovery_count']==1
    assert clean.manifest['excluded_lanes']==[1,2]


def test_unknown_reference_identity_cannot_claim_holdout_independence():
    g,d=fixture();bank=TaskSpaceReferenceBankV1.from_packed(d,g)
    del bank.entries[0]['lane_id']
    with pytest.raises(ValueError,match='identity missing'):
        bank.assert_excludes([2])


def test_toi_surrogate_uses_racket_at_arrival_without_ball_gradient():
    from quadsmash.rl.supermimic_toi_racket_objective import intercept_loss
    geometry, data=fixture()
    bank=TaskSpaceReferenceBankV1.from_packed(data,geometry)
    before=data['state'][3].clone().requires_grad_()
    after=before.detach().clone().requires_grad_()
    start=torch.tensor([.03,.03,.03]);end=start+.01
    ref=bank.bind(before,'approach',geometry=geometry,elapsed=start,require_toi=True)
    ref['intercept']=dict(time=torch.tensor([.035,.041,float('inf')]),
                          valid=torch.tensor([True,True,False]),position=before.detach()[:,49:52]+.05)
    loss,weights=intercept_loss(before,after,start,end,ref,torch.ones(3,dtype=torch.bool),torch.ones(3),geometry)
    assert weights.tolist()==[1,0,0]
    for grad in torch.autograd.grad(loss,(before,after)):
        assert grad[0,:49].norm()>0 and grad[:,49:].count_nonzero()==0
        assert grad[1:].count_nonzero()==0

    fixed=torch.tensor([.7,0.,0.])
    _,w=intercept_loss(before,after,start,end,ref,torch.ones(3,dtype=torch.bool),
                       torch.tensor([.1,1.,1.]),geometry,fixed_weights=fixed)
    torch.testing.assert_close(w,fixed)
    _,lost=intercept_loss(before,after,start,end,ref,torch.zeros(3,dtype=torch.bool),
                          torch.ones(3),geometry,fixed_weights=fixed)
    assert not ((fixed<=0)|(lost>0)).all()


def test_substep_target_joins_preceding_transition_preserves_legacy_tracking():
    geometry, data = fixture()
    old = TaskSpaceReferenceBankV1.from_packed(data, geometry)
    data['native_toi_racket'] = torch.full((12, 3, 13), float('nan'))
    measured = torch.tensor([.045, .01, .02, .03, 1., 2., 3., 1., 0., 0., 4., 5., 6.])
    data['native_toi_racket'][4, 0] = measured
    exact = TaskSpaceReferenceBankV1.from_packed(data, geometry, use_substep_toi=True)
    old.extend(exact)
    assert len(old.entries) == 2
    entry = old.entries[0]
    assert entry['trajectory_digest'] != entry['digest']
    torch.testing.assert_close(entry['toi_racket'], measured[1:])
    query = dict(geometry=geometry, elapsed=torch.tensor([.03]))
    assert old.bind(data['state'][3,:1], 'approach', **query)['anchor_time'].item() == pytest.approx(.05)
    assert old.bind(data['state'][3,:1], 'approach', **query, require_toi=True)['anchor_time'].item() == pytest.approx(.045)
    data['native_toi_racket'][4,0] = float('nan')
    missing = TaskSpaceReferenceBankV1.from_packed(data, geometry, use_substep_toi=True)
    assert 'toi_racket' not in missing.entries[0]
    data['native_toi_racket'][4,0] = measured
    data['native_toi_racket'][4,0,0] = .07
    with pytest.raises(ValueError, match='Invalid native'):
        TaskSpaceReferenceBankV1.from_packed(data, geometry, use_substep_toi=True)


def test_empty_toi_intervals_skip_backward_only_after_last_active_state():
    from quadsmash.rl.supermimic_toi_racket_objective import intercept_loss
    geometry, data = fixture()
    bank = TaskSpaceReferenceBankV1.from_packed(data, geometry)
    outputs = []
    for sparse in (False, True):
        initial = data['state'][3].clone().requires_grad_()
        state = initial
        clock = torch.full((len(state),), .03)
        ref = bank.bind(initial, 'approach', geometry=geometry, elapsed=clock, require_toi=True)
        ref['intercept'] = dict(time=torch.tensor([.045, .055, .091]), valid=torch.ones(3, dtype=torch.bool),
                                position=state.detach()[:,49:52]+.05)
        states, losses, weights = [], [], []
        for _ in range(5):
            after = state + .001
            loss, w = intercept_loss(state, after, clock, clock+.01, ref, torch.ones(3, dtype=torch.bool),
                                     torch.ones(3), geometry, detach_empty=sparse)
            states.append(after)
            losses.append(loss)
            weights.append(w)
            state, clock = after, clock+.01
        derivative = torch.autograd.grad(sum(losses), [initial]+states, allow_unused=True)
        outputs.append((torch.stack(losses).detach(), torch.stack(weights), derivative))
    dense, sparse = outputs
    torch.testing.assert_close(dense[:2], sparse[:2], rtol=0, atol=0)
    for a, b in zip(dense[2], sparse[2]):
        torch.testing.assert_close(a, b if b is not None else torch.zeros_like(a), rtol=0, atol=0)
    assert sparse[2][0].norm() > 0 and sparse[2][-1] is None
