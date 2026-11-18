"""Read-only complete native Recovery response comparison; no acceptance bounds."""
from pathlib import Path
import hashlib,json,torch
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_probe_region_v1 import contact_mode_groups
root=Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
probe=Path(json.loads((root/'recovery_probe_location.json').read_text())['path'])
output=Path(__file__).parent
if (output/'comparison.json').exists():raise FileExistsError(output/'comparison.json')
torch.set_num_threads(2)
d,provenance=load_response_data(probe,root/'fixed_split.json')
selection=json.loads((probe/'predeclared_selection.json').read_text())
modes={}
for branch in ('nominal_a','nominal_b','minus','plus'):
 r=torch.load(probe/branch/'native_probe.pt',map_location='cpu',weights_only=True)['raw']
 modes[branch]=[dict(foot_contact=r['next_feet'][i].bool().tolist(),body_ground_seen=bool(r['recovery_after']['body_ground_seen'][i])) for i in range(128)]
groups=contact_mode_groups(modes,128,'recovery')
command=d['target_plus']-d['target_minus'];truth=d['next_plus']-d['next_minus']
other=[d[k] for k in ('previous_action','feet','present','elapsed','launch')]
paths={'original':Path('/tmp/supermimic_racket_dynamics_20261114_YRFhKS/run/world.pt'),'response_calibrated':root/'native_response_fit_01/world.pt'}
models={}
for name,path in paths.items():
 c=torch.load(path,map_location='cpu',weights_only=True)
 model=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']);model.load_state_dict(c['model'],strict=True);model.eval().requires_grad_(False)
 with torch.no_grad():
  prediction={b:model(d['state'],d['target_'+b],*other)[0] for b in ('nominal_a','minus','plus')}
  nominal_feet=model(d['state'],d['target_nominal_a'],*other)[1].sigmoid()
 target=((d['target_minus']+d['target_plus'])/2).clone().requires_grad_(True)
 state=model(d['state'],target,*other)[0]
 jac=torch.stack([torch.autograd.grad(state[:,10+j].sum(),target,retain_graph=True)[0] for j in range(3)],1)
 projected=(jac*command[:,None]).sum(-1).detach()
 response=prediction['plus']-prediction['minus'];rows=[]
 for i in range(128):
  item=dict(lane=i,heldout=i in provenance['validation_lanes'],first_touchdown_in_source=i in selection['first_touchdown_lanes'],mode_group=groups[i],final_command_delta_rad=float(command[i].norm()),fields={})
  for label,sl in [('root_angular_velocity',slice(10,13)),('joint_position',slice(13,31)),('joint_velocity',slice(31,49))]:
   a=truth[i,sl];b=response[i,sl];an=float(a.norm());bn=float(b.norm())
   item['fields'][label]=dict(native_delta=a.tolist(),prediction_delta=b.tolist(),native_norm=an,prediction_norm=bn,relative_error=float((a-b).norm()/an) if an>1e-7 else None,cosine=float((a*b).sum()/(an*bn)) if an>1e-7 and bn>1e-7 else None)
  a=truth[i,10:13];g=projected[i];an=float(a.norm());gn=float(g.norm())
  item['projected_root_jacobian']=g.tolist()
  item['jacobian_cosine']=(float((a*g).sum()/(an*gn)) if an>1e-7 and gn>1e-7 and groups[i]=='smooth-mode' else None)
  item['nominal_angular_velocity_error_norm']=float((prediction['nominal_a'][i,10:13]-d['next_nominal_a'][i,10:13]).norm())
  rows.append(item)
 models[name]=dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,finite=bool(torch.isfinite(projected).all() and all(torch.isfinite(p).all() for p in prediction.values())))
result=dict(status='MEASURED_NOT_ADMITTED',provenance=provenance,selection=selection,mode_counts={k:groups.count(k) for k in set(groups)},models=models,actor_updates=0,world_updates=0,critic_updates=0,formal_evaluation=False,scope='Same-prefix native final-command response. Mode switches are finite differences, not smooth Jacobians; unknown native substep cache/modes not reconstructed. One direction/time is not full Recovery qualification. No limits inferred from results.')
(output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],mode_counts=result['mode_counts'],heldout_first_touchdown={name:[x for x in m['rows'] if x['lane'] in selection['heldout_first_touchdown_lanes']] for name,m in models.items()}),indent=2))
