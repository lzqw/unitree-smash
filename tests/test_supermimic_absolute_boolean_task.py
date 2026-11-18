import pytest
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import (
    NativeTaskTransitionSurrogateV1,AbsoluteBooleanTaskSurrogateV2,task_model_for_schema)


def schema():
    return TaskMemorySchemaV1([MemoryField('rally/contact',(),'bool',1),MemoryField('rally/time',(),'float',1)])


def test_explicit_schema_retains_v1_persistence_and_v2_absolute_logits():
    old=task_model_for_schema(NativeTaskTransitionSurrogateV1.schema,schema(),18,hidden=(8,8))
    new=task_model_for_schema(AbsoluteBooleanTaskSurrogateV2.schema,schema(),18,hidden=(8,8))
    new.load_state_dict(old.state_dict(),strict=True)
    values=torch.tensor([[0.,2.],[1.,3.]])
    output=torch.zeros_like(values,requires_grad=True)
    assert torch.allclose(old.boolean_logits(values,output)[:,0],torch.tensor([-9.21024,9.21007]),atol=1e-4)
    assert torch.equal(new.boolean_logits(values,output),output)
    assert torch.equal(torch.autograd.grad(new.boolean_logits(values,output).sum(),output)[0],torch.ones_like(output))
    with pytest.raises(ValueError,match='Unknown'):
        task_model_for_schema('missing',schema(),18)


def test_migration_initializes_only_boolean_output_rows_from_known_training_labels():
    m=AbsoluteBooleanTaskSurrogateV2(schema(),18,hidden=(8,8))
    before={k:v.clone() for k,v in m.state_dict().items()}
    values=torch.tensor([[0.,2.],[1.,3.],[1.,4.],[0.,0.]])
    valid=torch.tensor([[1.,1.],[1.,1.],[1.,1.],[0.,0.]])
    metadata=m.initialize_boolean_outputs(values,valid)
    assert metadata['known_training_counts']==[3.]
    assert metadata['boolean_frequency'][0]==pytest.approx(2/3)
    assert torch.count_nonzero(m.net.net[-1].weight[0])==0
    assert m.net.net[-1].bias[0].sigmoid().item()==pytest.approx(2/3)
    head_name=next(name for name,module in m.named_modules() if module is m.net.net[-1])
    for name,after in m.state_dict().items():
        if name in (head_name+'.weight',head_name+'.bias'):
            assert torch.equal(after[1:],before[name][1:])
        else:assert torch.equal(after,before[name])


def test_unknown_boolean_labels_are_not_silently_treated_as_false():
    m=AbsoluteBooleanTaskSurrogateV2(schema(),18,hidden=(8,8))
    with pytest.raises(ValueError,match='no known'):
        m.initialize_boolean_outputs(torch.zeros(2,2),torch.zeros(2,2))
    with pytest.raises(ValueError,match='native binary'):
        m.initialize_boolean_outputs(torch.full((2,2),.5),torch.ones(2,2))
