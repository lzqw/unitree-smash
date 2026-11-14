import torch
from quadsmash.rl.supermimic_physical_imagination_adapter_v1 import matrix_quaternion
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1, GROUPS
from quadsmash.rl.supermimic_task_transition_model_v1 import NativeTaskTransitionSurrogateV1


def test_rotation_quaternion_finite_gradient_and_sign_equivalence():
    from quadsmash.rl.supermimic_full_body_geometry_v1 import rotation as quaternion_matrix
    q=torch.tensor([[1.,.1,.2,.3],[.01,1.,.2,.1],[.1,.2,1.,.3],[.1,.3,.2,1.]])
    q=torch.nn.functional.normalize(q,dim=-1).requires_grad_(True)
    rotation=quaternion_matrix(q)
    rebuilt=matrix_quaternion(rotation)
    torch.testing.assert_close(quaternion_matrix(rebuilt),rotation,atol=1e-6,rtol=1e-6)
    grad=torch.autograd.grad(rebuilt.square().mul(torch.arange(4)).sum(),q)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0


def test_task_surrogate_preserves_unset_timestamps_and_input_gradient():
    memory={}
    for path in GROUPS:
        child=memory
        for part in path.split('/'):child=child.setdefault(part,{})
        child['clock']=torch.tensor([float('nan'),0.])
        child['flag']=torch.tensor([False,True])
    memory['paid_events']=torch.zeros(2,8)
    schema=TaskMemorySchemaV1.from_native(memory,2)
    model=NativeTaskTransitionSurrogateV1(schema,18,hidden=(16,16))
    values,valid=schema.encode(memory);model.set_normalization(values,valid)
    model.requires_grad_(False)
    state=torch.randn(2,55,requires_grad=True)
    targets=torch.randn(2,18,requires_grad=True)
    out=model(state,state+.01,targets,torch.zeros(2,18),torch.ones(2,4),torch.zeros(2),memory)
    assert torch.isnan(out['memory']['rally']['clock'][0])
    grad=torch.autograd.grad(out['hazards'].sum()+out['numeric'].sum(),(state,targets))
    assert all(torch.isfinite(x).all() and x.norm()>0 for x in grad)
    assert all(p.grad is None for p in model.parameters())
    assert not memory['paid_events'].any()
