import torch
from quadsmash.tasks.stand.observation_contract import (PolicyHistory,FRAME_SIZE,OBSERVATION_SIZE,
    FRAME_FIELDS,relative_velocity)

def test_history_is_once_per_tick_and_partial_reset_isolated():
    h=PolicyHistory(3,'cpu')
    a=torch.arange(3*FRAME_SIZE).reshape(3,FRAME_SIZE).float()
    h.observe(a,0)
    h.observe(a+1,0)
    assert torch.equal(h.frames[:,0],a)
    h.observe(a+2,1)
    assert torch.equal(h.frames[:,1],a)
    other=h.frames[1:].clone()
    h.reset(torch.tensor([0]))
    h.observe(a+3,1)
    assert torch.equal(h.frames[1:],other)
    assert torch.count_nonzero(h.frames[0,1:])==0

def test_rotating_reference_point_velocity_correction():
    q=torch.tensor([[1.,0,0,0]])
    point=torch.tensor([[1.,0,0]])
    velocity=torch.tensor([[0.,2.,0]])
    result=relative_velocity(q,point,velocity,torch.zeros(1,3),torch.zeros(1,3),torch.tensor([[0.,0,2.]]))
    assert torch.allclose(result,torch.zeros(1,3))
    assert sum(FRAME_FIELDS.values())==FRAME_SIZE
    assert OBSERVATION_SIZE==794
