"""Compare diagnostic feedback to the actual schema, values and gradients."""
import importlib.util
import argparse
import json
import sys
from pathlib import Path
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import NativeTaskTransitionSurrogateV1

torch.set_num_threads(2)
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=root/'task_boolean_repair_01/feedback_roundtrip_contract.json')
parser.add_argument('--candidate-module',type=Path)
args=parser.parse_args();path=args.output
if path.exists():raise FileExistsError(path)
p=torch.load('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt',map_location='cpu',weights_only=True);d=p['tensors']
schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in p['task_schema']['fields']])
m=NativeTaskTransitionSurrogateV1(schema,d['torque'].shape[-1])
m.load_state_dict(torch.load(root/'task_boolean_repair_01/task.pt',map_location='cpu',weights_only=True)['model'],strict=True)
lanes=torch.tensor(json.loads((root/'fixed_split.json').read_text())['validation_lanes'])
t=torch.tensor([255,271,301,499]).repeat_interleave(len(lanes));l=lanes.repeat(4)
out=m.forward_values(d['state'][t,l],d['next_state'][t,l],d['targets'][t,l],d['previous_action'][t,l],d['feet'][t,l],d['elapsed'][t,l],d['values'][t,l],d['valid'][t,l])
raw=out['numeric'];valid=out['valid_logits'].sigmoid()
actual=schema.encode(schema.decode_surrogate(raw,valid,p['template']))
spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).with_name('audit_task_event_rollouts.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
mirror=module.adapter_feedback(schema,raw,valid,d['values'][t,l],d['next_state'][t,l],d['final_action'][t,l],d['previous_action'][t,l],d['present'][t,l],d['elapsed'][t,l],t*160,None,2.,.01,['decode'])
for a,b in zip(actual,mirror):torch.testing.assert_close(a,b,rtol=0,atol=0)
a=torch.autograd.grad(actual[0].sum(),raw,retain_graph=True)[0]
b=torch.autograd.grad(mirror[0].sum(),raw)[0]
torch.testing.assert_close(a,b,rtol=0,atol=0)
candidate_result=None
if args.candidate_module:
    spec=importlib.util.spec_from_file_location('quadsmash.rl._feedback_real_audit',args.candidate_module)
    candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
    candidate_schema=candidate.TaskMemorySchemaV1(schema.fields)
    proposed=candidate_schema.feedback_values(raw,valid)
    for x,y in zip(proposed,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)
    g=torch.autograd.grad(proposed[0].sum(),raw)[0]
    torch.testing.assert_close(g,a,rtol=0,atol=0)
    candidate_result=dict(value_max_difference=0,input_gradient_max_difference=0,
                          candidate_module=str(args.candidate_module),applied_to_production=False)
fields=[];offset=0
for f in schema.fields:
    sl=slice(offset,offset+f.width);offset+=f.width
    fields.append(dict(path=f.path,kind=f.kind,numeric_max_difference=float((raw[:,sl]-actual[0][:,sl]).detach().abs().max()),validity_max_difference=float((valid[:,sl]-actual[1][:,sl]).detach().abs().max())))
result=dict(real_states=len(t),decode_roundtrip_max_difference=0,decode_roundtrip_gradient_max_difference=0,
    raw_vs_execution_numeric_max=float((raw-actual[0]).detach().abs().max()),raw_vs_execution_validity_max=float((valid-actual[1]).detach().abs().max()),
    fields=fields,candidate=candidate_result,actor_updates=0,model_updates=0,scope='Exact existing schema decode/encode parity; recorded states, no native step or optimizer')
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='fields'},indent=2))
print('largest changed values',sorted(fields,key=lambda x:x['numeric_max_difference'],reverse=True)[:5])
