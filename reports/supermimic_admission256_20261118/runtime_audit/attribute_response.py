"""Read-only response attribution; native next robot is an ORACLE, never rollout input."""
import json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features

torch.set_num_threads(2)
root=Path(__file__).resolve().parent
runs={k:torch.load(root/k/'native_probe.pt',weights_only=True,map_location='cpu') for k in ('minus','plus')}
r=runs['minus'];sl=slice(r['spec']['lane'],r['spec']['lane']+1);p=r['raw']['start']['before']['physical'];state=encode_physical(p)[sl]
base=PhysicalWorldV2(r['task_contract']['asset']['sha256']).eval();base.load_state_dict(torch.load('/tmp/supermimic_racket_dynamics_20261114_YRFhKS/run/world.pt',weights_only=True,map_location='cpu')['model'])
model=ActionConditionedBallV1().eval();model.load_state_dict(torch.load(root.parent/'action_ball_neighborhood/action_ball_candidate.pt',weights_only=True,map_location='cpu')['model'])
case=json.loads(Path(r['task_contract']['suite']['path']).read_text())['cases'][r['spec']['lane']]
launch=torch.tensor([case['launch']['position_m']+case['launch']['velocity_mps']])
with torch.no_grad():
 prior,_=base.prior(state,p['ball_present'][sl],p['elapsed_s'][sl],launch)
 predicted={};native={};responses={};details={}
 for name,r in runs.items():
  pred,_,_=base(state,r['raw']['final_joint_targets'][sl],p['final_action'][sl],p['foot_contact'][sl].float(),p['ball_present'][sl],p['elapsed_s'][sl],launch)
  predicted[name]=pred;native[name]=r['raw']['next_state'][sl]
  for label,next_robot in (('model',pred),('native_next_robot_oracle',native[name])):
   value,detail=model(state,next_robot,prior[:,49:],base.geometry,p['ball_present'][sl],details=True)
   responses.setdefault(label,{})[name]=value[:,3:]
   details.setdefault(label,{})[name]={k:v.tolist() for k,v in detail.items()}
 delta_native=native['plus']-native['minus'];delta_model=predicted['plus']-predicted['minus']
 metrics={}
 for name,start,stop in [('root_position',0,3),('root_angular_velocity',10,13),('joint_position',13,31),('joint_velocity',31,49)]:
  a=delta_native[:,start:stop];b=delta_model[:,start:stop]
  metrics[name]={'native_norm':float(a.norm()),'model_norm':float(b.norm()),'error_norm':float((a-b).norm()),'native':a.tolist(),'model':b.tolist()}
 response={label:(branches['plus']-branches['minus']).tolist() for label,branches in responses.items()}
 result={'scope':'single calibration case root-cause attribution only; oracle prohibited in deployment/WM forward','native_velocity_delta':delta_native[:,52:55].tolist(),'robot_response':metrics,'ball_response':response,'sweep_details':details,'optimizer_updates':0}
 with (root/'response_attribution.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='sweep_details'},indent=2))
