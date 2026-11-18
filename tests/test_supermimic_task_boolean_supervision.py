import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1, MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import NativeTaskTransitionSurrogateV1


def model():
    schema=TaskMemorySchemaV1((MemoryField('rally/hit',(),'bool',1),
                              MemoryField('rally/time',(),'float',1)))
    return NativeTaskTransitionSurrogateV1(schema,18,hidden=(8,8))


def test_native_boolean_flip_has_unsaturated_logit_gradient():
    m=model();logit=torch.tensor([[-9.21024,0.]],requires_grad=True)
    out=dict(boolean_logits=logit,numeric=torch.cat((logit[:,:1].sigmoid(),logit[:,1:]),-1))
    loss=m.memory_supervision_loss(out,torch.tensor([[1.,0.]]),torch.ones(1,2))
    gradient=torch.autograd.grad(loss,logit)[0]
    assert gradient[0,0]<-.99
    assert gradient[0,1]==0


def test_unknown_boolean_is_not_a_negative_example():
    m=model();logit=torch.tensor([[5.,0.]],requires_grad=True)
    out=dict(boolean_logits=logit,numeric=logit.sigmoid())
    loss=m.memory_supervision_loss(out,torch.zeros(1,2),torch.tensor([[0.,0.]]))
    assert loss==0
    assert torch.equal(torch.autograd.grad(loss,logit)[0],torch.zeros_like(logit))


def test_real_valued_memory_keeps_physical_normalization():
    m=model();m.memory_scale.copy_(torch.tensor([.01,2.]))
    x=torch.tensor([[0.,4.]],requires_grad=True)
    loss=m.memory_supervision_loss(dict(numeric=x,boolean_logits=x),torch.zeros_like(x),torch.tensor([[0.,1.]]))
    assert loss==4
    assert torch.equal(torch.autograd.grad(loss,x)[0],torch.tensor([[0.,2.]]))


def test_extra_logits_do_not_change_existing_forward_or_parameter_schema():
    m=model();m.normalization_ready.fill_(True)
    m.net.requires_grad_(False)
    for p in m.net.parameters():p.zero_()
    out=m.forward_values(torch.zeros(2,55),torch.zeros(2,55),torch.zeros(2,18),torch.zeros(2,18),torch.zeros(2,4),torch.zeros(2),torch.tensor([[0.,3.],[1.,4.]]),torch.ones(2,2))
    assert torch.equal(out['numeric'][:,1],torch.tensor([3.,4.]))
    assert torch.allclose(out['numeric'][:,0],torch.tensor([.0001,.9999]))
    assert torch.equal(out['numeric'][:,0],out['boolean_logits'][:,0].sigmoid())
    restored=model();restored.load_state_dict(m.state_dict(),strict=True)
