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
