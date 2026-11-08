import torch
from scripts.native_landing_contact_observer_v2 import reduce_pairs


def test_ragged_contacts_do_not_mix_sensor_pairs():
    values=torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
    counts=torch.tensor([[2],[0],[1]],dtype=torch.int64)
    starts=torch.tensor([[0],[999999],[3]],dtype=torch.int64)
    result=reduce_pairs(values,counts,starts)
    assert torch.equal(result,torch.tensor([[5.,7.,9.],[0.,0.,0.],[10.,11.,12.]]))


def test_full_patch_capacity_keeps_all_force_components():
    values=torch.arange(24,dtype=torch.float32).reshape(8,3)
    result=reduce_pairs(values,torch.tensor([8]),torch.tensor([0]))
    assert torch.equal(result[0],values.sum(0))
