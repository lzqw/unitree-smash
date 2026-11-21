#!/usr/bin/env python3
"""Compare measured-patch sticking and sliding responses on held multistep G07 executions."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn.functional import normalize

from audit_supermimic_native_window_response import complete_window
from audit_supermimic_paired_body_transfer import response_metrics
from audit_supermimic_pilot_friction_anchors import sticking_delta
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,sha
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch,_same_contract
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap,round_gap
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate_inverse
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def patch_friction(arms, surfaces, velocity, spin, normals, budget, inertia, *, reverse=False):
    """Project each observed anchor onto its own tangent plane and friction disk."""
    result = torch.zeros_like(arms)
    indices = range(len(arms)-1,-1,-1) if reverse else range(len(arms))
    for _ in range(4):
        for j in indices:
            n, arm = normals[j], arms[j]
            slip = velocity + torch.linalg.cross(spin,arm) - surfaces[j]
            tangent_slip = slip - n * torch.dot(slip,n)
            # In the tangent plane the sphere's effective mass is diagonal along
            # the projected arm and its perpendicular, including off-centre anchors.
            projected_arm = arm - n*torch.dot(n,arm)
            response = 1 + arm.square().sum()/inertia
            along = torch.dot(tangent_slip,projected_arm)*projected_arm/(inertia + torch.dot(n,arm).square())
            trial = result[j] - (tangent_slip + along)/response
            next_value = trial * (budget[j]/trial.norm().clamp_min(1.e-12)).clamp(max=1.)
            change = next_value-result[j]
            result[j] = next_value
            velocity = velocity+change
            spin = spin+torch.linalg.cross(arm,change)/inertia
    return result.sum(0)


def independent_tangent_rows(arms, surfaces, velocity, spin, normals, budget, inertia, rel_linear, iterations):
    """Prescribed-surface approximation with independent native-style tangent response multipliers."""
    tangent = rel_linear - (normals * rel_linear).sum(-1,keepdim=True) * normals
    axis = torch.zeros_like(normals)
    use_x = normals[:,0].abs() < 2**-.5
    axis[use_x,0] = 1.
    axis[~use_x,2] = 1.
    fallback = torch.linalg.cross(axis,normals)
    u = normalize(torch.where(tangent.square().sum(-1,keepdim=True) > .0001,tangent,fallback),dim=-1)
    basis = torch.stack((u,torch.linalg.cross(normals,u)),dim=1)
    lever = torch.linalg.cross(arms[:,None].expand_as(basis),basis)
    inverse_response = (1. + lever.square().sum(-1)/inertia).reciprocal()
    result = torch.zeros_like(arms)
    for _ in range(iterations):
        for j,arm in enumerate(arms):
            slip = velocity + torch.linalg.cross(spin,arm) - surfaces[j]
            delta = (basis[j] @ slip) * inverse_response[j]
            trial = result[j] - delta @ basis[j]
            updated = trial * (budget[j]/trial.norm().clamp_min(1.e-12)).clamp(max=1.)
            change = updated-result[j]
            result[j] = updated
            velocity = velocity+change
            spin = spin+torch.linalg.cross(arm,change)/inertia
    return result.sum(0)


def friction_trace(trace,geometry,mat,mass):
    t=trace['tensors']
    scale=trace['physics_dt_s']/mass
    nlanes=t['shuttle_state_w'].shape[1]
    modes=('pre','pre_radial','post_racket','patch','patch_reversed','row_single','row_iterated','actual')
    result={m:torch.zeros(nlanes,3,dtype=torch.float64) for m in modes}
    rounded=(geometry.racket_round_centers,geometry.racket_round_axes,geometry.racket_round_halves,
             geometry.racket_round_radii,geometry.racket_round_capsule)
    records=[]
    for i,(nf,ff) in enumerate(zip(trace['manifold_frames'][1:],trace['racket_friction_frames'][1:],strict=True)):
        ids=(ff['force_w_n'].norm(dim=-1)>.001)
        if not ids.any():continue
        for lane in ff['lane'][ids].unique().tolist():
            f=ids&(ff['lane']==lane)
            force=ff['force_w_n'][f].double()*scale
            true=force.sum(0)
            result['actual'][lane]+=true
            use=(nf['lane']==lane)&(nf['filter_index']==0)&(nf['normal_force_n'].abs()>.001)
            if not use.any():
                records.append({'lane':lane,'step':i,'normal_points':0,'friction_points':len(force)})
                continue
            state=t['shuttle_state_w'][i,lane].double()
            center,v,omega=state[:3],state[7:10],state[10:13]
            inertia=.4*geometry.radius**2
            anchors=ff['point_w_m'][f].double()
            points=nf['point_w'][use].double()
            loads=nf['normal_force_n'][use,None].double()*nf['normal_w'][use].double()*scale
            load=loads.sum(0)
            normal=normalize(load,dim=-1)
            arms=anchors-center
            spin_n=torch.linalg.cross(points-center,loads).sum(0)/inertia
            rc=t['racket_position_w'][i,lane].double()
            q=t['racket_quaternion_w'][i,lane].double().expand(len(points),-1)
            local=_quat_rotate_inverse(q,points-rc)
            ds=torch.cat((box_gap(local,geometry.racket_box_centers,geometry.racket_box_halves),
                          round_gap(local,*rounded)),-1)
            shape=ds.abs().argmin(-1)
            friction_budget=(loads.norm(dim=-1)*mat[shape,1]).sum()
            by_mode={}
            for mode in ('pre','pre_radial','post_racket'):
                lever=-geometry.radius*normal[None] if mode=='pre_radial' else arms
                when=i+(mode=='post_racket')
                surface=t['racket_linear_velocity_w'][when,lane].double() + torch.linalg.cross(
                    t['racket_angular_velocity_w'][when,lane].double().expand_as(lever),center+lever-t['racket_position_w'][when,lane].double())
                slip=v+load+torch.linalg.cross((omega+spin_n).expand_as(lever),lever)-surface
                dv=sticking_delta(lever,slip,normal,inertia,friction_budget)
                result[mode][lane]+=dv
                by_mode[mode]=dv.tolist()
            # Native force records identify the racket link, but not a collision shape.
            # Bind measured anchors to nearest measured normal points in world space.
            distances = torch.cdist(anchors, points)
            nearest = distances.argmin(dim=1)
            patch_ids = shape[nearest]
            budget = torch.zeros_like(arms[:,0])
            for shape_id in patch_ids.unique():
                group = patch_ids==shape_id
                budget[group] = loads[shape==shape_id].norm(dim=-1).sum()*mat[shape_id,1]/group.sum()
            patch_normals = normalize(loads[nearest],dim=-1)
            surface = t['racket_linear_velocity_w'][i,lane].double() + torch.linalg.cross(
                t['racket_angular_velocity_w'][i,lane].double().expand_as(arms),anchors-rc)
            for mode in ('patch','patch_reversed'):
                dv = patch_friction(arms,surface,v+load,omega+spin_n,patch_normals,budget,inertia,
                                    reverse=mode=='patch_reversed')
                result[mode][lane]+=dv
                by_mode[mode]=dv.tolist()
            for mode,iterations in [('row_single',1),('row_iterated',4)]:
                dv = independent_tangent_rows(arms,surface,v+load,omega+spin_n,patch_normals,budget,inertia,
                                             v-t['racket_linear_velocity_w'][i,lane].double(),iterations)
                result[mode][lane]+=dv
                by_mode[mode]=dv.tolist()
            records.append(dict(lane=lane,step=i,shapes=shape.tolist(),normal_points=len(points),
                                friction_points=len(force),anchor_to_normal_distance_m=distances.amin(dim=1).tolist(),
                                inferred_anchor_shapes=patch_ids.tolist(),measured=true.tolist(),bound_mps=float(friction_budget),
                                normalized_friction=float(true.norm()/load.norm().clamp_min(1.e-12)),predicted=by_mode))
    return result,records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','response','material','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    r=read(a.response);status=read(a.collection/'status.json')
    if status['status']!='COLLECTION_FINISHED' or sha(a.collection/'status.json')!=r['status_sha256'] or sha(a.material)!=r['material_sha256']:
        raise ValueError('Physical evidence or material changed')
    mat=torch.tensor(validate_contract(read(a.material),r['task_contract']['asset']['sha256'])['effective'])
    geometry=BallImpulsePriorV1().double().preview
    phys=Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json'
    if sha(phys)!=geometry.geometry_hashes['configs/shuttle_physics_v2.json']:raise ValueError('Changed mass or radius')
    mass=read(phys)['mass_kg']
    runs={}
    for run in status['completed']:
        b=run['branch'];folder=a.collection/run['spec']
        if sha(folder/b/'native_probe.pt')!=run['probe_sha256']:raise ValueError('Changed native branch')
        runs[b]=_load_branch(folder,b)
    _same_contract(runs)
    summaries={};frames={}
    for b,run in runs.items():
        sums={};events=[]
        for i,tr in enumerate(complete_window(run['raw'],run['spec'],160)):
            data,raw=friction_trace(tr,geometry,mat,mass)
            for k,v in data.items():sums[k]=sums.get(k,0.)+v
            events.extend(dict(row,control_tick=run['spec']['action_tick']+i,
                               lane=run['spec']['lanes'][row['lane']]) for row in raw)
        summaries[b]=sums;frames[b]=events
    native=summaries['plus']['actual']-summaries['minus']['actual']
    repeat=summaries['nominal_b']['actual']-summaries['nominal_a']['actual']
    stats={}
    for g in ['all',*sorted({c['group'] for c in r['cases']})]:
        sel=torch.tensor([g=='all' or c['group']==g for c in r['cases']])
        stats[g]={k:response_metrics(native[sel],(v-summaries['minus'][k])[sel],repeat[sel])
                  for k,v in summaries['plus'].items() if k!='actual'}
    with a.output.open('x') as out:
        json.dump(dict(groups=stats,frames=frames,source_sha256=sha(a.response),
                  scope='Signed native friction only; post-racket velocities and loaded anchor positions used for attribution',
                  contact_gradients_admitted=False),out,indent=2,allow_nan=False)
        out.write('\n')


if __name__=='__main__':main()
