"""Task-only64-step recurrence on heldout measured physical transitions.

Future physical values here are teacher-forcing diagnostics, not an imagined
actor path. Task memory is initialized once and then never read from labels.
"""
from pathlib import Path
import argparse,json,torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import NativeTaskTransitionSurrogateV1
from quadsmash.rl.supermimic_event_hazard_v1 import advance_event_latches
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows

torch.set_num_threads(2)
parser=argparse.ArgumentParser();parser.add_argument('--lead',type=int,default=1);args=parser.parse_args()
if not 1<=args.lead<64:raise ValueError('Lead must place the measured event inside the64-step window')
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
outpath=root/f'task_boolean_repair_01/contact_rollout_lead{args.lead:02d}_audit.json'
if outpath.exists():raise FileExistsError(outpath)
package=torch.load('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt',map_location='cpu',weights_only=True);d=package['tensors']
schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in package['task_schema']['fields']])
m=NativeTaskTransitionSurrogateV1(schema,d['torque'].shape[-1]);m.load_state_dict(torch.load(root/'task_boolean_repair_01/task.pt',map_location='cpu',weights_only=True)['model'],strict=True);m.eval()
lanes=json.loads((root/'fixed_split.json').read_text())['validation_lanes'];valid=valid_windows(d['episode'],d['terminated'],d['timeout'],64)
edge=(d['next_event_state'][:,:,1]>d['event_state'][:,:,1])|(d['next_event_state'][:,:,6]>d['event_state'][:,:,6])
pairs=[]
for lane in lanes:
 for tick in edge[:,lane].nonzero().flatten().tolist():
  t=max(0,tick-args.lead)
  if t<len(valid) and valid[t,lane]:pairs.append((t,lane))
t,l=torch.tensor(pairs).unbind(1)
indices={};offset=0
for f in schema.fields:indices[f.path]=offset;offset+=f.width
had=indices['rally/had_contact'];released=indices['rally/released']
values=d['values'][t,l];validity=d['valid'][t,l];paid=d['event_state'][t,l];predicted=[];truth=[];events=[]
with torch.no_grad():
 for j in range(64):
  at=t+j
  out=m.forward_values(d['state'][at,l],d['next_state'][at,l],d['targets'][at,l],d['previous_action'][at,l],d['feet'][at,l],d['elapsed'][at,l],values,validity)
  values,validity=out['numeric'],out['valid_logits'].sigmoid();paid=advance_event_latches(paid,out['hazards'])
  predicted.append(values[:,[had,released]].clone());truth.append(d['next_values'][at,l][:,[had,released]]);events.append(paid.clone())
predicted=torch.stack(predicted);truth=torch.stack(truth);events=torch.stack(events)
rows=[]
for i,(start,lane) in enumerate(pairs):
 pg=(predicted[:,i]>.5).all(-1);tg=(truth[:,i]>.5).all(-1)
 pi=pg.nonzero().flatten();ti=tg.nonzero().flatten()
 rows.append(dict(lane=lane,start=start,predicted_first_gate_tick=start+int(pi[0]) if len(pi) else None,true_first_gate_tick=start+int(ti[0]) if len(ti) else None,
                  gate_disagreement_ticks=int((pg!=tg).sum()),predicted_gate_flags=predicted[:,i].tolist(),predicted_paid_events=events[:,i].tolist()))
result=dict(scope='Task-only64-step recurrence with recorded physical transitions; no optimizer, no native/whole-model success claim',lead=args.lead,rows=rows,actor_updates=0,model_updates=0)
outpath.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(lead=args.lead,count=len(rows),matched=sum(r['predicted_first_gate_tick']==r['true_first_gate_tick'] for r in rows),missed=sum(r['predicted_first_gate_tick'] is None for r in rows),gate_disagreement_ticks=sum(r['gate_disagreement_ticks'] for r in rows))))
