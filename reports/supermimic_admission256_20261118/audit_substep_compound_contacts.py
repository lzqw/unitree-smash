"""Observed impact geometry diagnostic; not a causal policy/model input."""
import argparse,json
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap,round_gap,BallGeometryPreviewV1
from quadsmash.rl.supermimic_full_body_geometry_v1 import rotation
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import box_normal,round_normal
p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
torch.set_num_threads(2);r=torch.load(a.probe,map_location='cpu',weights_only=True);x=r['raw']['substep_trace']['tensors'];g=BallGeometryPreviewV1();cfg=json.loads(Path('configs/shuttle_physics_v2.json').read_text());dt=cfg['native_physics_dt_s'];rows=[]
for lane in range(128):
 touch=x['racket_force_w'][:,lane].norm(dim=-1)>cfg['contact_force_threshold_n'];ids=(touch[1:]&~touch[:-1]).nonzero().flatten()+1
 if not len(ids):continue
 i=int(ids[0]);old=x['shuttle_state_w'][i-1,lane];v=old[7:10];incoming=v+dt*(v.new_tensor([0.,0.,-9.81])-cfg['quadratic_drag_k_per_m']*v.norm()*v);free=old[:3]+dt*incoming
 variants={}
 for mode,index in [('pre',i-1),('post',i)]:
  R=rotation(x['racket_quaternion_w'][index,lane:lane+1])[0];local=(R.T@(free-x['racket_position_w'][index,lane]))[None]
  gaps=torch.cat((box_gap(local,g.racket_box_centers,g.racket_box_halves),round_gap(local,g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,g.racket_round_radii,g.racket_round_capsule)),-1)[0]-g.radius
  normals=torch.cat((box_normal(local,g.racket_box_centers,g.racket_box_halves),round_normal(local,g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,g.racket_round_radii,g.racket_round_capsule)),-2)[0]
  ids=gaps.argsort()[:8];nb=len(g.racket_box_centers)
  variants[mode]=dict(local=local[0].tolist(),within_original_contact_offsets=int((gaps<=2*cfg['contact_offset_m']).sum()),nearest=[dict(part=int(j),kind='box' if int(j)<nb else 'round',gap=float(gaps[j]),normal=(R@normals[j]).tolist()) for j in ids])
 force=x['racket_force_w'][i,lane];ray=force/force.norm()
 rows.append(dict(lane=lane,index=i,frame_hit=bool(r['raw']['task_after']['frame_hit'][lane]),force_direction_not_contact_normal=ray.tolist(),variants=variants))
a.output.write_text(json.dumps(dict(scope=__doc__,rows=rows),indent=2)+'\n')
for row in rows:print(row['lane'],row['variants']['pre']['within_original_contact_offsets'],row['variants']['post']['within_original_contact_offsets'],[(z['part'],round(z['gap'],6)) for z in row['variants']['post']['nearest'][:3]])
