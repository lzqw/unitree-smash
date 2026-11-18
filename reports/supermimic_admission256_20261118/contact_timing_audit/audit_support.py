"""Read-only recorded-contact support diagnosis; no optimizer or native run."""
from pathlib import Path
import json,torch
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_wm_validation_v1 import event_aligned_windows
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
output=Path(__file__).parent
packed=Path('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt')
torch.set_num_threads(2)
package=torch.load(packed,map_location='cpu',weights_only=True);data=package['tensors']
checkpoint=torch.load(root/'native_response_fit_01/world.pt',map_location='cpu',weights_only=True)
base=PhysicalWorldV2(checkpoint['source_manifest']['task_contract']['asset']['sha256']);base.load_state_dict(checkpoint['model'],strict=True)
ball=BallResidualV2();ball.load_state_dict(torch.load('/tmp/supermimic_ball_sequence_20261114_axsonnit/run/ball.pt',map_location='cpu',weights_only=True)['model'],strict=True)
sweep=ActionConditionedBallV1();sweep.load_state_dict(torch.load(root/'native_response_fit_01/action_ball_candidate.pt',map_location='cpu',weights_only=True)['model'],strict=True)
base.eval();ball.eval();sweep.eval()
suite=Path(checkpoint['source_manifest']['task_contract']['suite']['path']);cases=json.loads(suite.read_text())['cases'];launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases])
validation=set(json.loads((root/'fixed_split.json').read_text())['validation_lanes'])
windows=event_aligned_windows(data,range(128),horizon=1,offsets=(-1,0,1),maximum=1000)
rows=[]
with torch.no_grad():
 for off,item in windows.items():
  t,l=item['starts'].unbind(1);s=data['state'][t,l];truth=data['next_state'][t,l]
  other=[data[k][t,l] for k in ('previous_action','feet','present','elapsed')]
  pred,_,_=base(s,data['targets'][t,l],*other,launch[l]);prior,_=base.prior(s,data['present'][t,l],data['elapsed'][t,l],launch[l])
  features=ball_geometry_features(s,base.geometry);prob,response=ball.components(features)
  correction,details=sweep(s,pred,prior[:,49:],base.geometry,data['present'][t,l],details=True)
  legacy=(prob[:,1:,None]*response).sum(1)
  dense={}
  for name,robot in [('predicted',pred),('oracle_robot_diagnostic',truth)]:
   start=features[:,6:9];end=ball_geometry_features(torch.cat((robot[:,:49],prior[:,49:]),-1),base.geometry)[:,6:9]
   alpha=torch.linspace(0,1,161)
   points=start[:,None,:]+(end-start)[:,None,:]*alpha[None,:,None]
   allg=sweep.preview.racket_gap(points)
   dense[name]=dict(minimum=allg.min(-1).values,three=allg[:,[0,80,160]].min(-1).values,index=allg.argmin(-1))
  for j in range(len(t)):
   rows.append(dict(lane=int(l[j]),tick=int(t[j]),offset=int(off),heldout=int(l[j]) in validation,legacy_mode_probabilities=prob[j].tolist(),legacy_ball_delta=legacy[j].tolist(),sweep_ball_delta=correction[j].tolist(),needed_ball_delta=(truth[j,49:]-prior[j,49:]).tolist(),support=float(details['support'][j]),gaps=details['gaps'][j].tolist(),dense={k:{f:float(v[j]) for f,v in z.items()} for k,z in dense.items()}))
summary={}
for off in (-1,0,1):
 arr=[x for x in rows if x['offset']==off];summary[str(off)]={}
 for group in ('all','heldout'):
  x=[q for q in arr if group=='all' or q['heldout']];n=len(x)
  summary[str(off)][group]=dict(count=n,legacy_racket_probability_mean=sum(z['legacy_mode_probabilities'][1] for z in x)/n,action_support_mean=sum(z['support'] for z in x)/n,
    predicted_three_misses_dense_overlap=sum(z['dense']['predicted']['three']>0 and z['dense']['predicted']['minimum']<=0 for z in x),oracle_three_misses_dense_overlap=sum(z['dense']['oracle_robot_diagnostic']['three']>0 and z['dense']['oracle_robot_diagnostic']['minimum']<=0 for z in x),
    predicted_dense_overlap=sum(z['dense']['predicted']['minimum']<=0 for z in x),oracle_dense_overlap=sum(z['dense']['oracle_robot_diagnostic']['minimum']<=0 for z in x))
result=dict(summary=summary,rows=rows,scope='Dense interpolation/oracle robot is diagnostic only, not true PhysX substep trajectory, not policy forward. No optimizer steps, no native score, no threshold modification.')
(output/'support_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
