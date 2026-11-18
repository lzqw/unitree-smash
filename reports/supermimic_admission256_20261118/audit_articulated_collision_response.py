"""Measured impulse / authored articulated inertia attribution, diagnostic only.
No MPC solver/controller is used. Native force and contact point are hindsight
labels, never candidate rollout inputs. Authored armature/drive approximation
is explicitly excluded; this tests mechanical coupling, not model admission.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1,rotation

p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(2)
r=torch.load(a.probe,map_location='cpu',weights_only=True);x=r['raw']['substep_trace']['tensors'];dt=r['raw']['substep_trace']['physics_dt_s']
rigid=LoadedPinocchioModelV1();geometry=FullBodyGeometryV1(rigid.record['asset_sha256'],dtype=torch.float64);rid=geometry.ids['Racket']
threshold=float(json.loads(Path('configs/shuttle_physics_v2.json').read_text())['contact_force_threshold_n'])
rows=[]
for lane in range(128):
 mask=x['racket_force_w'][:,lane].norm(dim=-1)>threshold
 ids=(mask[1:]&~mask[:-1]).nonzero().flatten()+1
 if not len(ids):continue
 i=int(ids[0]);qj=x['joint_position'][i-1,lane].double()[None]
 zero=torch.zeros(1,3,dtype=torch.float64);identity=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
 rel=geometry(qj,torch.zeros_like(qj),zero,identity,zero,zero)
 Rbefore=rotation(x['racket_quaternion_w'][i-1,lane:lane+1].double())[0].numpy()
 root_rotation=Rbefore@rel['rotation'][0,rid].numpy().T
 q=pin.neutral(rigid.model);q[3:7]=pin.Quaternion(root_rotation).coeffs();q[rigid.q_indices]=qj[0].numpy()
 M=pin.crba(rigid.model,rigid.data,q);M=np.triu(M)+np.triu(M,1).T
 pin.computeJointJacobians(rigid.model,rigid.data,q);pin.updateFramePlacements(rigid.model,rigid.data)
 J=pin.getFrameJacobian(rigid.model,rigid.data,rigid.body_frames['Racket'],pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
 point=x['racket_contact_position_w'][i,lane].double().numpy()
 if not np.isfinite(point).all():raise ValueError('Observed contact lacks its point')
 arm=point-x['racket_position_w'][i-1,lane].double().numpy()
 Jpoint=J[:3]+np.cross(J[3:].T,arm).T
 A=Jpoint@np.linalg.solve(M,Jpoint.T)
 impulse=x['racket_force_w'][i,lane].double().numpy()*dt
 predicted=-A@impulse
 def surface(index):
  R=rotation(x['racket_quaternion_w'][index,lane:lane+1].double())[0].numpy()
  rpoint=R@Rbefore.T@arm
  return x['racket_linear_velocity_w'][index,lane].double().numpy()+np.cross(x['racket_angular_velocity_w'][index,lane].double().numpy(),rpoint)
 observed=surface(i)-surface(i-1)
 background=surface(i-1)-surface(i-2) if i>1 else None
 corrected=observed-background if background is not None else observed
 norm=np.linalg.norm(observed);pnorm=np.linalg.norm(predicted)
 row=dict(lane=lane,index=i,frame_hit=bool(r['raw']['task_after']['frame_hit'][lane]),
  observed_surface_delta=observed.tolist(),predicted_impulse_surface_delta=predicted.tolist(),
  background_previous_substep_delta=None if background is None else background.tolist(),
  error_norm=float(np.linalg.norm(predicted-observed)),background_corrected_error_norm=float(np.linalg.norm(predicted-corrected)),
  cosine=float(observed@predicted/(norm*pnorm)) if norm>1e-9 and pnorm>1e-9 else None,
  effective_inverse_mass=A.tolist(),mass_min_eigenvalue=float(np.linalg.eigvalsh(M).min()),
  native_force_impulse=impulse.tolist())
 rows.append(row);print(json.dumps({k:row[k] for k in ('lane','frame_hit','error_norm','background_corrected_error_norm','cosine')}))
a.output.write_text(json.dumps(dict(scope=__doc__,rows=rows,model_asset_sha256=rigid.record['asset_sha256'],source_model_sha256=rigid.sha256,all_optimizer_updates=0),indent=2)+'\n')
