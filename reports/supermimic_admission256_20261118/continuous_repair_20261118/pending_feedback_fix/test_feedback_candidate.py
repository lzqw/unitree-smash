"""Isolated tests of a staged patch; production source is not modified."""
import importlib.util
import sys
from pathlib import Path
import pytest
import torch

path=Path(__file__).parent/'quadsmash/rl/supermimic_task_memory_schema_v1.py'
spec=importlib.util.spec_from_file_location('quadsmash.rl._staged_task_feedback',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)


def schema():
    return module.TaskMemorySchemaV1([
        module.MemoryField('rally/time',(), 'float',1),
        module.MemoryField('rally/velocity',(2,), 'float',2),
        module.MemoryField('rally/contact',(), 'bool',1),
        module.MemoryField('rally/count',(), 'int',1)])


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('batch',[1,128])
def test_candidate_matches_real_decode_encode_values_and_input_gradient(dtype,batch):
    contract=schema()
    values=torch.tensor([[2.,3.,4.,.2,7.]],dtype=dtype).repeat(batch,1).requires_grad_()
    valid=torch.tensor([[.49,.5,.51,.01,.01]],dtype=dtype).repeat(batch,1)
    template={'rally':{}}
    expected=contract.encode(contract.decode_surrogate(values,valid,template))
    actual=contract.feedback_values(values,valid)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
    a=torch.autograd.grad(actual[0].square().sum(),values,retain_graph=True)[0]
    b=torch.autograd.grad(expected[0].square().sum(),values)[0]
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert torch.isfinite(a).all()
    assert (a[:,:2]==0).all() and (a[:,2:]!=0).all()


def test_missing_float_never_becomes_observed_zero():
    values=torch.zeros(2,5)
    valid=torch.ones(2,5);valid[0,0]=.49
    result,mask=schema().feedback_values(values,valid)
    assert result[0,0]==result[1,0]==0
    assert mask[0,0]==0 and mask[1,0]==1


def test_finite_boolean_and_counter_use_existing_decode_validity_semantics():
    values=torch.tensor([[2.,3.,4.,1.2,7.]])
    valid=torch.full_like(values,.0001)
    result,mask=schema().feedback_values(values,valid)
    assert result.tolist()==[[0.,0.,0.,1.,7.]]
    assert mask.tolist()==[[0.,0.,0.,1.,1.]]


@pytest.mark.parametrize('values,valid',[(torch.zeros(1,4),torch.zeros(1,4)),
    (torch.zeros(1,5),torch.zeros(2,5)),(torch.zeros(5),torch.zeros(5))])
def test_wrong_feedback_shape_is_rejected(values,valid):
    with pytest.raises(ValueError,match='feedback width'):
        schema().feedback_values(values,valid)
