"""Read-only heldout native memory-transition diagnostics; no optimizer."""
import argparse,json,hashlib
from pathlib import Path
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import NativeTaskTransitionSurrogateV1

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(2)
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
data_path=Path('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt')
package=torch.load(data_path,map_location='cpu',weights_only=True);d=package['tensors']
c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in package['task_schema']['fields']])
m=NativeTaskTransitionSurrogateV1(schema,d['torque'].shape[-1]);m.load_state_dict(c['model'],strict=True);m.eval()
lanes=json.loads((root/'fixed_split.json').read_text())['validation_lanes']
assert lanes==c['config']['validation_lanes']
pred=[]
with torch.no_grad():
 for t in range(800):
  out=m.forward_values(d['state'][t,lanes],d['next_state'][t,lanes],d['targets'][t,lanes],d['previous_action'][t,lanes],d['feet'][t,lanes],d['elapsed'][t,lanes],d['values'][t,lanes],d['valid'][t,lanes])
  pred.append(out['numeric'])
prediction=torch.stack(pred);offset=0;rows=[]
for field in schema.fields:
 columns=slice(offset,offset+field.width);offset+=field.width
 if field.kind!='bool':continue
 before=d['values'][:,lanes,columns];target=d['next_values'][:,lanes,columns];valid=d['next_valid'][:,lanes,columns].bool();prob=prediction[:,:,columns]
 rise=(target>.5)&(before<.5)&valid;fall=(target<.5)&(before>.5)&valid;steady=(before==target)&valid
 rows.append(dict(path=field.path,rise_count=int(rise.sum()),rise_detected=int((rise&(prob>=.5)).sum()),rise_probability_mean=float(prob[rise].mean()) if rise.any() else None,
                  fall_count=int(fall.sum()),fall_detected=int((fall&(prob<.5)).sum()),steady_wrong=int((steady&((prob>=.5)!=(target>=.5))).sum()),steady_count=int(steady.sum())))
result=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),scope='unchanged32 heldout lanes, actual current+next physical state, task memory one-step forecast; no rollout or native success claim',model_updates=0,actor_updates=0,fields=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([row for row in rows if any(s in row['path'] for s in ('had_contact','released','body_ground','rally_recovery','contact_seen','airborne_seen','string_hit'))],indent=2))
