"""Read-only timing diagnosis; recorded physical transitions are teacher forced."""
import argparse,hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import task_model_for_schema
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_event_hazard_v1 import advance_event_latches
from reports.supermimic_admission256_20261118.audit_task_event_rollouts import adapter_feedback

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(2)
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
package=torch.load('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt',map_location='cpu',weights_only=True);d=package['tensors']
schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in package['task_schema']['fields']])
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
model=task_model_for_schema(checkpoint['schema'],schema,d['torque'].shape[-1]);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
world=PhysicalWorldV2(package['source_manifest']['task_contract']['asset']['sha256']).eval()
fields={};offset=0
for f in schema.fields:fields[f.path]=slice(offset,offset+f.width);offset+=f.width
keys=('auxiliary/contact_recovery/stable_run','auxiliary/contact_recovery/longest_stable','auxiliary/contact_recovery/qualified_landing','recovery/rally_recovery_success','recovery/body_ground_seen')
split=json.loads((root/'fixed_split.json').read_text());dt=.01;substeps=round(dt/package['source_manifest']['task_contract']['physics_dt_s']);results={}
with torch.no_grad():
 for group in ('train_lanes','validation_lanes'):
  rows=[(int(t)-16,l,int(t)) for l in split[group] for t in (d['next_event_state'][:,l,4]>d['event_state'][:,l,4]).nonzero().flatten() if int(t)>=16]
  if not rows:results[group]=[];continue
  start,l,event=torch.tensor(rows).unbind(1);values=d['values'][start,l];validity=d['valid'][start,l];paid=d['event_state'][start,l];output=[dict(lane=int(lane),event_tick=int(tick),steps=[]) for _,lane,tick in rows]
  for j in range(33):
   at=start+j
   if not torch.equal(d['episode'][at,l],d['episode'][start,l]) or (d['terminated'][at-1,l]|d['timeout'][at-1,l]).any():raise ValueError('Diagnostic would cross completed episode')
   args=[d[k][at,l] for k in ('state','next_state','targets','previous_action','feet','elapsed')]
   out=model.forward_values(*args,values,validity)
   native=model.forward_values(*args,d['values'][at,l],d['valid'][at,l])
   before=values
   values,validity=adapter_feedback(schema,out['numeric'],out['valid_logits'].sigmoid(),values,d['next_state'][at,l],d['final_action'][at,l],d['previous_action'][at,l],d['present'][at,l],d['elapsed'][at,l],at*substeps+package['template']['auxiliary']['timing']['_sim_step_counter'],world.geometry,world.launch_s,dt,('decode','clock','command','launch','geometry'))
   paid=advance_event_latches(paid,out['hazards'])
   for i,row in enumerate(output):
    row['steps'].append(dict(relative_tick=j-16,hazard_with_predicted_memory=float(out['hazards'][i,4].sigmoid()),hazard_with_native_memory=float(native['hazards'][i,4].sigmoid()),predicted_paid=float(paid[i,4]),native_paid=float(d['next_event_state'][at[i],l[i],4]),memory={key:dict(predicted_before=before[i,fields[key]].tolist(),native_before=d['values'][at[i],l[i],fields[key]].tolist(),native_valid=d['valid'][at[i],l[i],fields[key]].tolist()) for key in keys}))
  results[group]=output
r=dict(scope=__doc__,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),actor_updates=0,model_updates=0,results=results,limitation='Not a native success score, no whole-world free-running accuracy claim; validation examples are diagnostics only.')
a.output.write_text(json.dumps(r,indent=2)+'\n')
for group,rows in results.items():
 pre=[s for r in rows for s in r['steps'] if s['relative_tick']<0];true=[r['steps'][16] for r in rows]
 print(group,'events',len(rows),'mean_pre_hazard_native',sum(s['hazard_with_native_memory'] for s in pre)/len(pre),'mean_event_hazard_native',sum(s['hazard_with_native_memory'] for s in true)/len(true))
