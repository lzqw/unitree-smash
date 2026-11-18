"""Diagnostic ONLY: replace predicted robot endpoints by recorded endpoints.

This is an oracle error attribution, never the deployed/imagined forward.
Native future endpoints remain supervision/evaluation evidence only.
"""
import argparse,json,hashlib
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_action_ball_v1 import action_ball_from_checkpoint

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();output=a.run/'endpoint_sensitivity_attribution.json'
if output.exists():raise FileExistsError(output)
torch.set_num_threads(2)
config=json.loads((a.run/'fit/config.json').read_text());models={k:torch.load(a.run/'fit'/f,map_location='cpu',weights_only=True) for k,f in [('world','world.pt'),('ball','ball.pt'),('action','action_ball_candidate.pt')]}
w=PhysicalWorldV2(models['world']['source_manifest']['task_contract']['asset']['sha256']);w.load_state_dict(models['world']['model'],strict=True)
b=BallResidualV2();b.load_state_dict(models['ball']['model'],strict=True)
s=action_ball_from_checkpoint(models['action']);s.load_state_dict(models['action']['model'],strict=True)
w=w.to(a.device);b=b.to(a.device);s=s.to(a.device)
inventory=json.loads((a.run/'collected_response_inventory.json').read_text());result=[]
def comparison(native,predicted):
 an=float(native.norm());bn=float(predicted.norm())
 return dict(native_norm=an,predicted_norm=bn,error_norm=float((native-predicted).norm()),cosine=float((native*predicted).sum()/(an*bn)) if an>1e-7 and bn>1e-7 else None)
with torch.no_grad():
 for source in inventory['sources']:
  pair,prov=load_response_data(Path(source['path']),Path(config['argv'][config['argv'].index('--split-manifest')+1]))
  pair={k:v.to(a.device) for k,v in pair.items()}
  prior,_=w.prior(pair['state'],pair['present'],pair['elapsed'],pair['launch'])
  prob,_=b.components(ball_geometry_features(pair['state'],w.geometry));kwargs={'racket_probability':prob[:,1]} if getattr(s,'requires_racket_probability',False) else {}
  predictions={};oracles={};robots={}
  for branch in ('minus','plus'):
   robots[branch]=w(pair['state'],pair['target_'+branch],pair['previous_action'],pair['feet'],pair['present'],pair['elapsed'],pair['launch'])[0]
   predictions[branch]=s(pair['state'],robots[branch],prior[:,49:],w.geometry,pair['present'],**kwargs)
   oracles[branch]=s(pair['state'],pair['next_'+branch],prior[:,49:],w.geometry,pair['present'],**kwargs)
  truth=pair['next_plus'][:,52:]-pair['next_minus'][:,52:]
  predicted=predictions['plus'][:,3:]-predictions['minus'][:,3:];oracle=oracles['plus'][:,3:]-oracles['minus'][:,3:]
  record=dict(path=source['path'],head=source['head'],tick=source['tick'],groups={})
  for group in ('train_lanes','validation_lanes'):
   rows=[]
   for lane in source['groups'][group]['nominal_contact_lanes']:
    rows.append(dict(lane=lane,actual_model=comparison(truth[lane],predicted[lane]),recorded_endpoint_diagnostic=comparison(truth[lane],oracle[lane]),joint_position_response=comparison(pair['next_plus'][lane,13:31]-pair['next_minus'][lane,13:31],robots['plus'][lane,13:31]-robots['minus'][lane,13:31]),joint_velocity_response=comparison(pair['next_plus'][lane,31:49]-pair['next_minus'][lane,31:49],robots['plus'][lane,31:49]-robots['minus'][lane,31:49])))
   record['groups'][group]=rows
  result.append(record)
output.write_text(json.dumps(dict(scope=__doc__,candidate_schema=s.schema,model_hashes={f:hashlib.sha256((a.run/'fit'/f).read_bytes()).hexdigest() for f in ('world.pt','ball.pt','action_ball_candidate.pt')},results=result,actor_updates=0,model_updates=0,not_native_admission=True),indent=2)+'\n')
for source in result:
 for r in source['groups']['validation_lanes']:
  print('tick',source['tick'],'lane',r['lane'],'model',r['actual_model']['cosine'],'native_endpoint',r['recorded_endpoint_diagnostic']['cosine'],'q',r['joint_position_response']['cosine'],'dq',r['joint_velocity_response']['cosine'])
