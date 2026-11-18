"""Hindsight collision-law attribution on real substeps, never rollout input.
Uses original geometry/restitution/friction. Future body pose and measured
contact point are diagnostic oracles, not causal model features or labels of
success. No policy, world model, or optimizer is modified.
"""
import argparse,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_full_body_geometry_v1 import rotation

p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(2)
r=torch.load(a.probe,map_location='cpu',weights_only=True);trace=r['raw']['substep_trace'];x=trace['tensors']
physics=json.loads(Path('configs/shuttle_physics_v2.json').read_text());model=BallImpulsePriorV1();dt=physics['native_physics_dt_s'];mu=physics['friction'];e=physics['restitution'];k=physics['quadratic_drag_k_per_m']
rows=[]
for lane in range(128):
 touch=x['racket_force_w'][:,lane].norm(dim=-1)>physics['contact_force_threshold_n']
 edges=touch[1:]&~touch[:-1]
 if not edges.any():continue
 i=int(edges.nonzero()[0])+1
 old=x['shuttle_state_w'][i-1,lane];v0=old[7:10];omega=old[10:13]
 accel=v0.new_tensor([0,0,-9.81])-k*v0.norm()*v0
 incoming=v0+dt*accel;freepos=old[:3]+dt*incoming
 true=x['shuttle_state_w'][i,lane,7:10]
 variants={}
 for mode in ('pre_pose','post_pose','sensor_point'):
  pose_index=i-1 if mode=='pre_pose' else i
  center=x['racket_position_w'][pose_index,lane];rot=rotation(x['racket_quaternion_w'][pose_index,lane:lane+1])[0]
  if mode=='sensor_point':
   if not x['racket_contact_position_valid'][i,lane]:continue
   point=x['racket_contact_position_w'][i,lane]
   ray=freepos-point;n=ray/ray.norm().clamp_min(1e-12);part=None
  else:
   local=(rot.T@(freepos-center))
   normal,part_tensor=model.racket_surface(local[None]);n=rot@normal[0];part=int(part_tensor[0])
   point=freepos-model.preview.radius*n
  arm=point-center
  surface=x['racket_linear_velocity_w'][i-1,lane]+torch.linalg.cross(x['racket_angular_velocity_w'][i-1,lane],arm)
  relative=incoming+torch.linalg.cross(omega,-model.preview.radius*n)-surface
  vn=(relative*n).sum();tangent=relative-vn*n
  dvn=-(1+e)*vn.clamp_max(0)*n
  speed=tangent.norm();dvt=torch.minimum((2./7.)*speed,mu*dvn.norm())*tangent/speed.clamp_min(1e-12)
  predicted=incoming+dvn-dvt
  variants[mode]=dict(normal=n.tolist(),part=part,surface_velocity=surface.tolist(),predicted_velocity=predicted.tolist(),
     error_norm=float((predicted-true).norm()),normal_relative_velocity=float(vn),normal_impulse_delta=dvn.tolist(),friction_delta=(-dvt).tolist())
 rows.append(dict(lane=lane,index=i,frame_hit=bool(r['raw']['task_after']['frame_hit'][lane]),native_before_velocity=v0.tolist(),native_after_velocity=true.tolist(),native_force=x['racket_force_w'][i,lane].tolist(),variants=variants))
a.output.write_text(json.dumps(dict(scope=__doc__,probe=str(a.probe),rows=rows,optimizer_updates=0),indent=2)+'\n')
for row in rows:print(row['lane'],row['frame_hit'],{k:round(v['error_norm'],6) for k,v in row['variants'].items()})
