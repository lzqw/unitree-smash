"""Exercise validation against the actual imagination memory round trip."""
from types import SimpleNamespace
import torch
import pytest
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1, MemoryField


def test_validation_replays_actual_imagination_memory(monkeypatch):
    from quadsmash.rl import supermimic_wm_validation_v1 as module
    n=2;t=64
    schema=TaskMemorySchemaV1([MemoryField('rally/time',(),'float',1),MemoryField('rally/contact',(),'bool',1)])
    emitted=torch.tensor([[2.,1.2],[3.,.4]])
    validity=torch.tensor([[.49,.01],[.51,.01]])
    expected=schema.encode(schema.decode_surrogate(emitted,validity,{'rally':{}}))
    seen=[]
    class Task:
        memory_schema=schema
        def forward_values(self,s,ns,u,prev,feet,elapsed,values,valid):
            seen.append((values.clone(),valid.clone()))
            return {'numeric':emitted,'valid_logits':torch.logit(validity),'hazards':torch.full((n,8),-80.)}
    adapter=SimpleNamespace(physical_model=SimpleNamespace(),ball_model=None,task_model=Task())
    state=torch.zeros(t,n,55);state[:,:,3]=1
    window=dict(state=state,next_state=state.clone(),targets=torch.zeros(t,n,18),previous_action=torch.zeros(t,n,18),
        feet=torch.zeros(t,n,4),next_feet=torch.zeros(t,n,4),present=torch.ones(t,n),elapsed=torch.arange(t)[:,None].expand(t,n)*.01,
        values=torch.zeros(t,n,2),valid=torch.ones(t,n,2),event_state=torch.zeros(t,n,8),next_event_state=torch.zeros(t,n,8))
    monkeypatch.setattr(module,'ball_transition',lambda base,ball,s,u,prev,feet,present,elapsed,launch,**kw:(s.clone(),feet.clone(),present.clone()))
    module.validate_windows(adapter,window,torch.zeros(n,6),horizons=(1,4,16,32,64))
    assert len(seen)==64
    for actual in seen[1:]:
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
