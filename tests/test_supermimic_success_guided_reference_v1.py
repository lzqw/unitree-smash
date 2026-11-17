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
