import torch

from quadsmash.rl.supermimic_actor_start_bank_v1 import ActorStartBank, CATEGORIES
from quadsmash.rl.supermimic_toi_racket_objective import bind_intercept_reference


def test_toi_bank_keeps_lane_and_category_identity_after_reference_filter():
    bank=ActorStartBank(seed=2)
    for index in range(2):
        masks={name:torch.zeros(3,dtype=torch.bool) for name in CATEGORIES}
        masks['gate_onset'][0]=True
        masks['pre_contact'][2]=True
        bank.entries.append(dict(start={'index':index},masks=masks,stream=3,policy_version=16,native_tick=100+index))
    calls=[]
    def allowed(row):
        calls.append(row['index'])
        return torch.tensor([False,False,row['index']==1])
    row,report=bank.sample(1,eligible_lanes=allowed)
    assert row['index']==1 and row['_actor_lane_mask'].tolist()==[False,False,True]
    assert report['requested']=='gate_onset' and report['actual']=='pre_contact' and report['shortfall']==1
    assert sorted(calls)==[0,1]
    assert all(entry['masks']['gate_onset'][0] for entry in bank.entries)
    row,report=bank.sample(1,eligible_lanes=lambda r:torch.zeros(3,dtype=torch.bool))
    assert row is None and report['actual'] is None


def test_toi_start_rejects_expired_arrival_and_unmatched_reference(monkeypatch):
    from types import SimpleNamespace
    from quadsmash.rl import supermimic_toi_racket_objective as toi
    state=SimpleNamespace(physical=torch.zeros(4,55),clock_goal_context={'elapsed':torch.ones(4)})
    forecast=dict(valid=torch.tensor([True,True,False,True]),time=torch.tensor([1.1,1.,1.2,1.1]))
    monkeypatch.setattr(toi,'flight_intercept',lambda state,horizon:forecast)
    def bind(*args,**kwargs):
        assert kwargs['require_toi'] and torch.equal(kwargs['event_time'],forecast['time'])
        return dict(matched=torch.tensor([True,True,True,False]))
    ref=bind_intercept_reference(state,SimpleNamespace(bind=bind),None,.64)
    assert ref['matched'].tolist()==[True,False,False,False]
