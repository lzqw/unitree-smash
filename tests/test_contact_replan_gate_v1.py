import torch
from quadsmash.control.contact_replan_gate_v1 import ContactReplanGate


def test_bounded_retry_preserves_verified_and_recovery_lanes_and_reset():
    gate=ContactReplanGate(3,'cpu');obs=torch.zeros(3,794);obs[:,67]=1.;obs[:,775]=2.
    valid=torch.tensor([False,True,False]);recovery=torch.tensor([False,False,True])
    assert gate.eligible(obs,valid,recovery).tolist()==[True,False,False]
    for k in range(4):
        obs[:,775]=2.+k*.16
        ids=torch.where(gate.eligible(obs,valid,recovery))[0]
        assert ids.tolist()==[0];gate.record(ids,obs[:,775])
        assert not gate.eligible(obs,valid,recovery).any()
    obs[:,775]=2.64
    assert not gate.eligible(obs,valid,recovery).any()
    gate.reset(torch.tensor([0]));obs[:,775]=2.
    assert gate.eligible(obs,valid,torch.zeros(3,dtype=torch.bool)).tolist()==[True,False,False]
    obs[0,773]=1.
    assert not gate.eligible(obs,valid,recovery).any()
    obs[0,773]=0.
    assert not gate.eligible(obs,valid,recovery).any()


def test_replan_window_is_not_extended_by_missing_ball():
    gate=ContactReplanGate(1,'cpu');obs=torch.zeros(1,794);obs[:,775]=2.7;obs[:,67]=1.
    assert not gate.eligible(obs,torch.tensor([False]),torch.tensor([False])).any()
