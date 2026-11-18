"""Read-only impact attribution; observed impact timing/normal are hindsight.
Original authored mass/inertia; friction by actual static binding paths, not fitted. Runtime PhysX material tensors remain to be read independently.
This is not an admitted rollout model, controller or performance evaluation.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,rotation
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(2)
r=torch.load(a.probe,map_location='cpu',weights_only=True);x=r['raw']['substep_trace']['tensors'];dt=r['raw']['substep_trace']['physics_dt_s']
rigid=LoadedPinocchioModelV1();geometry=FullBodyGeometryV1(rigid.record['asset_sha256'],dtype=torch.float64);rid=geometry.ids['Racket'];prior=BallImpulsePriorV1()
cfg=json.loads(Path('configs/shuttle_physics_v2.json').read_text());mu=cfg['friction'];e=cfg['restitution'];mass=cfg['mass_kg'];radius=prior.preview.radius
import ast
material_source=Path('/home/autodrive/Isaac/IsaacLab/source/isaaclab/isaaclab/sim/spawners/materials/physics_materials_cfg.py')
cls=next(x for x in ast.parse(material_source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='RigidBodyMaterialCfg')
defaults={x.target.id:ast.literal_eval(x.value) for x in cls.body if isinstance(x,ast.AnnAssign) and x.target.id in ('dynamic_friction','static_friction','friction_combine_mode')}
if defaults['friction_combine_mode']!='average':raise ValueError('Unsupported default combine contract')
frame_mu=(defaults['dynamic_friction']+mu)/2
rows=[]
for lane in range(128):
 mask=x['racket_force_w'][:,lane].norm(dim=-1)>cfg['contact_force_threshold_n'];ids=(mask[1:]&~mask[:-1]).nonzero().flatten()+1
 if not len(ids):continue
 i=int(ids[0]);qj=x['joint_position'][i-1,lane].double()[None];zero=torch.zeros(1,3,dtype=torch.float64)
 rel=geometry(qj,torch.zeros_like(qj),zero,torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64),zero,zero)
 Rbefore=rotation(x['racket_quaternion_w'][i-1,lane:lane+1].double())[0].numpy();rootR=Rbefore@rel['rotation'][0,rid].numpy().T
 q=pin.neutral(rigid.model);q[3:7]=pin.Quaternion(rootR).coeffs();q[rigid.q_indices]=qj[0].numpy()
 M=pin.crba(rigid.model,rigid.data,q);M=np.triu(M)+np.triu(M,1).T
 pin.computeJointJacobians(rigid.model,rigid.data,q);pin.updateFramePlacements(rigid.model,rigid.data)
 J=pin.getFrameJacobian(rigid.model,rigid.data,rigid.body_frames['Racket'],pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
 old=x['shuttle_state_w'][i-1,lane].double().numpy();v=old[7:10];omega=old[10:13]
 incoming=v+dt*(np.array([0,0,-9.81])-cfg['quadratic_drag_k_per_m']*np.linalg.norm(v)*v);freepos=old[:3]+dt*incoming
 actual=x['shuttle_state_w'][i,lane,7:10].double().numpy();variants={}
 for mode in ('pre_pose','post_pose','sensor_point'):
  index=i-1 if mode=='pre_pose' else i
  center=x['racket_position_w'][index,lane].double().numpy();R=rotation(x['racket_quaternion_w'][index,lane:lane+1].double())[0].numpy()
  if mode=='sensor_point':
   if not x['racket_contact_position_valid'][i,lane]:continue
   point=x['racket_contact_position_w'][i,lane].double().numpy();n=freepos-point;n=n/np.linalg.norm(n)
  else:
   local=torch.from_numpy(R.T@(freepos-center))[None];nl,part=prior.racket_surface(local);n=R@nl[0].numpy();point=freepos-radius*n
  # The material binding is on Collider (strings), not sibling frame/clamp shapes.
  if mode=='sensor_point':
   _,part=prior.racket_surface(torch.from_numpy(R.T@(freepos-center))[None])
  original_string_shapes=len(prior.preview.racket_box_centers)-1
  contact_mu=mu if int(part[0])<original_string_shapes else frame_mu
  arm=point-x['racket_position_w'][i-1,lane].double().numpy();Jp=J[:3]+np.cross(J[3:].T,arm).T;A=Jp@np.linalg.solve(M,Jp.T)
  surface=x['racket_linear_velocity_w'][i-1,lane].double().numpy()+np.cross(x['racket_angular_velocity_w'][i-1,lane].double().numpy(),arm)
  relative=incoming+np.cross(omega,-radius*n)-surface;vn=relative@n
  K=np.eye(3)/mass+2.5/mass*(np.eye(3)-np.outer(n,n))+A
  stick=np.linalg.solve(K,-relative-e*min(vn,0)*n);pn=stick@n;pt=stick-pn*n
  tangent=relative-vn*n;speed=np.linalg.norm(tangent)
  if vn>=0:impulse=np.zeros(3);regime='separating'
  elif pn>=0 and np.linalg.norm(pt)<=contact_mu*pn:impulse=stick;regime='sticking'
  else:
   direction=n-contact_mu*tangent/max(speed,1e-12);den=n@K@direction
   if den<=0:raise ValueError('Nonpositive effective mass')
   impulse=-(1+e)*vn/den*direction;regime='sliding'
  predicted=incoming+impulse/mass
  variants[mode]=dict(predicted_velocity=predicted.tolist(),error_norm=float(np.linalg.norm(predicted-actual)),regime=regime,effective_friction=contact_mu,normal=n.tolist(),normal_inverse_mass=float(n@A@n),impulse=impulse.tolist())
 rows.append(dict(lane=lane,index=i,frame_hit=bool(r['raw']['task_after']['frame_hit'][lane]),native_after_velocity=actual.tolist(),variants=variants))
a.output.write_text(json.dumps(dict(scope=__doc__,probe=str(a.probe),rows=rows,all_optimizer_updates=0,model_asset_sha256=rigid.record['asset_sha256']),indent=2)+'\n')
for row in rows:print(row['lane'],row['frame_hit'],{k:round(v['error_norm'],6) for k,v in row['variants'].items()})
