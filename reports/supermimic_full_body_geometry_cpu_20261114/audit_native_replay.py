"""Read-only full-model racket FK comparison; no native step or parameter fit."""
import hashlib,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,MODEL

torch.set_num_threads(2)
path=Path('/tmp/unitree_smash_g07_eventaware_long_20261114_094334/checkpoint.pt')
saved=torch.load(path,map_location='cpu',weights_only=False)
replay=saved['replay'];assert replay['ticks_written']<=replay['capacity']//replay['num_envs']
obs=replay['data']['obs'].flatten(0,1)
reset=Path('configs/full_racket_reset_v4.json')
default=torch.tensor(json.loads(reset.read_text())['arm_joint_positions_rad'])
m=FullBodyGeometryV1(json.loads(MODEL.read_text())['asset_sha256'])
racket=m.ids['Racket'];rows=[]
with torch.no_grad():
 for phase in range(5):
  ids=torch.where(obs[:,770:775].argmax(1)==phase)[0]
  if not len(ids):continue
  ids=ids[torch.linspace(0,len(ids)-1,min(2048,len(ids))).long()]
  x=obs[ids];n=len(x)
  q=torch.zeros(n,18);qd=torch.zeros_like(q)
  q[:,12:]=x[:,21:27]+default;qd[:,12:]=x[:,39:45]
  zero=torch.zeros(n,3);quat=torch.zeros(n,4);quat[:,0]=1
  result=m(q,qd,zero,quat,zero,zero)
  entry=dict(phase=phase,samples=n)
  for name,a,b in [('racket_position_b',result['position'][:,racket],x[:,45:48]),
      ('racket_normal_b',result['rotation'][:,racket,:,0],x[:,48:51]),
      ('racket_relative_velocity_b',result['velocity'][:,racket],x[:,51:54]),
      ('racket_relative_angular_velocity_b',result['angular_velocity'][:,racket],x[:,54:57])]:
   diff=(a-b).abs();entry[name]=dict(mae=float(diff.mean()),max_abs=float(diff.max()))
  rows.append(entry)
out=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_sha256=m.model_sha256,
    phases=rows,scope='Old-layout native replay, racket arm chain only; zero leg values do not affect root-mounted arm. No claim of foot or head native validation, full-world dynamics or policy success.')
Path(__file__).with_name('native_replay_geometry.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(rows))
