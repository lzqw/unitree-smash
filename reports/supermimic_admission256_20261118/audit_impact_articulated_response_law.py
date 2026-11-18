"""Read-only response attribution with original restitution and surface geometry.
Not an admitted model. No policy updates or native simulation. The endpoint
oracle is separately labelled and NEVER a candidate's causal forward input.
Coulomb sphere response uses original friction, first-contact zero incoming spin assumption and authored articulated effective inertia. Motor armature, implicit PD and support constraints are omitted. Diagnostic only.
"""
import argparse,json,hashlib
from pathlib import Path
import torch
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_physical_dataset_v1 import decode_physical
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_impact_timing_v1 import causal_sweep_time,interpolate_physical_state

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
out=a.run/'impact_articulated_response_law_diagnostic.json'
if out.exists():raise FileExistsError(out)
torch.set_num_threads(2)
c=torch.load(a.run/'fit/world.pt',map_location='cpu',weights_only=True)
w=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).cpu();w.load_state_dict(c['model'],strict=True);w.eval()
prior=BallImpulsePriorV1().cpu();mu=float(json.loads(Path('configs/shuttle_physics_v2.json').read_text())['friction']);config=json.loads((a.run/'fit/config.json').read_text())
split=Path(config['argv'][config['argv'].index('--split-manifest')+1])
inventory=json.loads((a.run/'collected_response_inventory.json').read_text())

def response(state,endpoint,present):
    free,_=w.prior(state,present,clock,launch)
    endpoint=torch.cat((endpoint[:,:49],free[:,49:]),-1)
    f0=ball_geometry_features(state,w.geometry);f1=ball_geometry_features(endpoint,w.geometry)
    time,timing=causal_sweep_time(f0[:,6:9],f0[:,9:12],f1[:,6:9],prior.preview)
    impact=interpolate_physical_state(state,endpoint,time,prior.dt)
    phys=decode_physical(impact)
    geo=w.geometry(phys['joint_position'],phys['joint_velocity'],phys['root_position_w'],phys['root_quaternion_w'],phys['root_link_velocity_w'],phys['root_angular_velocity_w'])
    rid=w.geometry.ids['Racket'];rot=geo['rotation'][:,rid];center=geo['position'][:,rid]
    pre_p,pre_v=prior.advance(state[:,49:52],state[:,52:55],time)
    local=(rot.transpose(-1,-2)@(pre_p-center)[...,None]).squeeze(-1)
    local_normal,part=prior.racket_surface(local)
    normal=(rot@local_normal[...,None]).squeeze(-1)
    arm=pre_p-center-prior.preview.radius*normal
    surface=geo['velocity'][:,rid]+torch.linalg.cross(geo['angular_velocity'][:,rid],arm)
    vn=((pre_v-surface)*normal).sum(-1)
    inverse,_=inertia(phys['joint_position'],phys['root_quaternion_w'],local-prior.preview.radius*local_normal)
    impulse_momentum,_=module.sphere_impulse(pre_v-surface,normal,inverse,mass=.0052,radius=prior.preview.radius,restitution=prior.restitution,friction=mu)
    impulse=impulse_momentum/.0052
    active=timing['crossing_found']&present.bool()
    impulse=torch.where(active[:,None],impulse,torch.zeros_like(impulse))
    post_p,post_v=prior.advance(pre_p,pre_v+impulse,prior.dt-time)
    return post_v,dict(time=time,active=active,normal=normal,surface_velocity=surface,part=part)

def metric(x,y):
    xn=float(x.norm());yn=float(y.norm())
    return dict(native_delta=x.tolist(),predicted_delta=y.tolist(),native_norm=xn,predicted_norm=yn,error_norm=float((x-y).norm()),cosine=float((x*y).sum()/(xn*yn)) if xn>1e-7 and yn>1e-7 else None)

import importlib.util
module_path=Path('reports/supermimic_admission256_20261118/continuous_repair_20261118/articulated_candidate/inertia_candidate.py')
spec=importlib.util.spec_from_file_location('inertia_candidate',module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
inertia=module.ArticulatedInertiaCandidate(w.geometry.asset_sha256,dtype=torch.float32)
results=[]
with torch.no_grad():
 for source in inventory['sources']:
    pairs,_=load_response_data(Path(source['path']),split);pairs={k:v.cpu() for k,v in pairs.items()}
    clock=pairs['elapsed'];launch=pairs['launch'];state=pairs['state']
    item=dict(path=source['path'],tick=source['tick'],head=source['head'],groups={})
    pred={};oracle={};info={}
    for branch in ('minus','plus'):
        endpoint=w(state,pairs['target_'+branch],pairs['previous_action'],pairs['feet'],pairs['present'],clock,launch)[0]
        pred[branch],info[branch]=response(state,endpoint,pairs['present'])
        oracle[branch],_=response(state,pairs['next_'+branch],pairs['present'])
    truth=pairs['next_plus'][:,52:55]-pairs['next_minus'][:,52:55]
    for group in ('train_lanes','validation_lanes'):
        rows=[]
        for lane in source['groups'][group]['nominal_contact_lanes']:
            rows.append(dict(lane=lane,causal=metric(truth[lane],pred['plus'][lane]-pred['minus'][lane]),
                oracle_endpoint=metric(truth[lane],oracle['plus'][lane]-oracle['minus'][lane]),
                diagnostics={b:{k:v[lane].tolist() for k,v in info[b].items()} for b in ('minus','plus')}))
        item['groups'][group]=rows
    results.append(item)
record=dict(scope=__doc__,results=results,actor_updates=0,model_updates=0,not_admission=True,world_sha256=hashlib.sha256((a.run/'fit/world.pt').read_bytes()).hexdigest())
out.write_text(json.dumps(record,indent=2)+'\n')
for s in results:
 for r in s['groups']['validation_lanes']:
    print(json.dumps(dict(tick=s['tick'],lane=r['lane'],causal=r['causal'],oracle=r['oracle_endpoint'])),flush=True)
