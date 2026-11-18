"""Recorded contact-time labels versus causal learned-robot sweep timing.

The timing estimate uses current relative position/velocity and the model's
endpoint only. Native contact timestamps are comparison labels, never inputs.
This read-only hard-crossing diagnostic does not change any model forward.
"""
import argparse,json,hashlib
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();dest=a.run/'causal_impact_timing_diagnostic.json'
if dest.exists():raise FileExistsError(dest)
torch.set_num_threads(2)
source=torch.load(a.run/'fit/world.pt',map_location='cpu',weights_only=True);world=PhysicalWorldV2(source['source_manifest']['task_contract']['asset']['sha256']);world.load_state_dict(source['model'],strict=True);preview=BallGeometryPreviewV1()
config=json.loads((a.run/'fit/config.json').read_text());split=Path(config['argv'][config['argv'].index('--split-manifest')+1]);inventory=json.loads((a.run/'collected_response_inventory.json').read_text());records=[]
with torch.no_grad():
 for item in inventory['sources']:
  if item['head']!='approach':continue
  directory=Path(item['path']);pair,prov=load_response_data(directory,split)
  current=ball_geometry_features(pair['state'],world.geometry)
  robot=world(pair['state'],pair['target_nominal_a'],pair['previous_action'],pair['feet'],pair['present'],pair['elapsed'],pair['launch'])[0]
  flight=world.prior(pair['state'],pair['present'],pair['elapsed'],pair['launch'])[0]
  future=ball_geometry_features(torch.cat((robot[:,:49],flight[:,49:]),-1),world.geometry)
  dt=preview.times[-1];time=preview.times[None,:,None]
  first=current[:,6:9];velocity=current[:,9:12];last=future[:,6:9]
  local=first[:,None]+time*velocity[:,None]+(time/dt).square()*(last-first-dt*velocity)[:,None]
  gap=preview.racket_gap(local);inside=gap<=0.;index=inside.long().argmax(1)
  hit=inside.any(1);estimate=preview.times[index]
  native=torch.load(directory/'nominal_a/native_probe.pt',map_location='cpu',weights_only=True)['raw'];clock=native['start']['before']['auxiliary']['environment']['ball_clock'];true=native['task_after']['contact_time']-clock
  for group in ('train_lanes','validation_lanes'):
   for lane in item['groups'][group]['nominal_contact_lanes']:
    tau=float(true[lane]);pred=float(estimate[lane]) if hit[lane] else None
    records.append(dict(tick=item['tick'],lane=lane,partition=group,native_event_delay_s=tau,predicted_first_geometry_crossing_s=pred,error_s=None if pred is None else pred-tau,gap_start=float(gap[lane,0]),minimum_gap=float(gap[lane].min()),native_model_time_different=bool(tau<0 or tau>float(dt)),quadratic_command_position_fraction=(tau/float(dt))**2))
result=dict(scope=__doc__,world_sha256=hashlib.sha256((a.run/'fit/world.pt').read_bytes()).hexdigest(),records=records,actor_updates=0,model_updates=0,limitation='Approximate current-frame quadratic relative motion, no PhysX trajectory reconstruction or independent contact proof.')
dest.write_text(json.dumps(result,indent=2)+'\n')
for g in ('train_lanes','validation_lanes'):
 rows=[r for r in records if r['partition']==g];e=[r['error_s'] for r in rows if r['error_s'] is not None]
 print(g,'count',len(rows),'geometry_no_crossing',sum(r['error_s'] is None for r in rows),'mean_abs_time_error_s',sum(abs(x) for x in e)/len(e) if e else None)
for row in records:
 if row['partition']=='validation_lanes':print(row)
