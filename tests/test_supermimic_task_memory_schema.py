import pytest
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,GROUPS


def fixture_memory():
    memory={}
    for path in GROUPS:
        value=memory
        parts=path.split('/')
        for part in parts:value=value.setdefault(part,{})
        value['clock']=torch.tensor([float('nan'),0.])
        value['bound']=torch.tensor([False,True])
    return memory


def test_unset_timestamp_cannot_alias_valid_zero():
    memory=fixture_memory();schema=TaskMemorySchemaV1.from_native(memory,2)
    values,valid=schema.encode(memory)
    assert torch.isfinite(values).all()
    assert not torch.equal(valid[0],valid[1])
    assert torch.isnan(memory['rally']['clock'][0])


def test_missing_memory_is_an_error_not_zero_fill():
    memory=fixture_memory();schema=TaskMemorySchemaV1.from_native(memory,2)
    del memory['auxiliary']['contact_recovery']['bound']
    with pytest.raises(KeyError):schema.encode(memory)


def test_packed_columns_preserve_native_precision_order_and_derivatives():
    memory=fixture_memory()
    first=torch.tensor([[1.,float('nan')],[float('inf'),2.]],requires_grad=True)
    second=torch.tensor([3.,4.],requires_grad=True)
    memory['rally'].update(a=first,aa=second,counter=torch.tensor([2**53+1,-1]),
                           double=torch.tensor([1.e40,float('-inf')],dtype=torch.float64))
    schema=TaskMemorySchemaV1.from_native(memory,2)
    actual,mask=schema.encode(memory)
    columns=[];masks=[]
    for field in schema.fields:
        value=memory
        for key in field.path.split('/'):
            value=value[key]
        valid=torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value,dtype=torch.bool)
        columns.append(torch.where(valid,value,0).reshape(2,-1).float())
        masks.append(valid.reshape(2,-1).float())
    expected=torch.cat(columns,-1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(mask,torch.cat(masks,-1))
    weights=torch.arange(actual.numel()).reshape_as(actual).float()
    for a,b in zip(torch.autograd.grad(actual,(first,second),weights),
                   torch.autograd.grad(expected,(first,second),weights),strict=True):
        assert torch.equal(a,b)
