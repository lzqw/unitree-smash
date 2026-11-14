"""CPU-only source-FK consistency diagnostic against stored native observations."""
import hashlib
import json
from pathlib import Path
import torch
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4

torch.set_num_threads(2)
path=Path('/tmp/unitree_smash_g07_eventaware_long_20261114_094334/checkpoint.pt')
saved=torch.load(path,map_location='cpu',weights_only=False)
replay=saved['replay']; data=replay['data']
assert replay['ticks_written'] <= replay['capacity']//replay['num_envs'], 'Handle wrapped replay explicitly'
obs=data['obs'].flatten(0,1)
reset_path=Path('configs/full_racket_reset_v4.json')
default=torch.tensor(json.loads(reset_path.read_text())['arm_joint_positions_rad'])
model=FullRacketKinematicsV4().float()
rows=[]
with torch.no_grad():
 for phase in range(5):
  ids=torch.where(obs[:,770:775].argmax(1)==phase)[0]
  if not len(ids):continue
  ids=ids[torch.linspace(0,len(ids)-1,min(len(ids),2048)).long()]
  x=obs[ids]; q=x[:,21:27]+default; dq=x[:,39:45]
  pos,normal,jac,_,_,axes,_=model(q,return_pose=True)
  vel=torch.einsum('bij,bj->bi',jac,dq)
  ang=torch.einsum('bji,bj->bi',axes,dq)
  result={'phase':phase,'samples':len(ids)}
  for name,pred,target in [('tool_position_b',pos,x[:,45:48]),('tool_normal_b',normal,x[:,48:51]),('tool_relative_velocity_b',vel,x[:,51:54]),('tool_relative_angular_velocity_b',ang,x[:,54:57])]:
   diff=(pred-target).abs();worst=int(diff.amax(1).argmax())
   result[name]={'mae':float(diff.mean()),'max_abs':float(diff.max()),'worst_flat_row':int(ids[worst]),'predicted':pred[worst].tolist(),'recorded':target[worst].tolist()}
  rows.append(result)
r={'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reset_config_sha256':hashlib.sha256(reset_path.read_bytes()).hexdigest(),'arm_default_rad':default.tolist(),'phases':rows,'scope':'Read-only recorded native observations from OLD layout training; tests kinematics only, not a new rollout, dynamics quality or complete policy reconstruction'}
Path(__file__).with_name('native_replay_geometry.json').write_text(json.dumps(r,indent=2)+'\n')
for row in rows: print(row['phase'],row['samples'],{k:v['max_abs'] for k,v in row.items() if isinstance(v,dict)})
