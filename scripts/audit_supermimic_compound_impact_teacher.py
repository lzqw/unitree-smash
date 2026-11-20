#!/usr/bin/env python3
"""Compare one- and multi-shape contact solves with incoming measured shuttle and racket states."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_supermimic_solver_action_response import BallImpulsePriorV1, BRANCHES, mode, sha
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap, round_gap
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import box_normal, round_normal
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate, _quat_rotate_inverse
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


def solve_patch(normals, mu, restitution, velocity, spin, surface, radius, active, normal_scale, iterations):
    """Projected sequential normal/tangent impulses on a sphere. Impulses are in m/s."""
    arm=-radius*normals
    support=normals.shape[1]
    normal_load=torch.zeros_like(mu)
    tangent_load=torch.zeros_like(normals)
    linear=velocity.clone()
    angular=spin.clone()
    vn=((velocity[:,None]+torch.cross(spin[:,None].expand_as(arm),arm,dim=-1)-surface)*normals).sum(-1)
    # Preserve the calibrated single-surface outgoing relative normal speed.
    target=vn+(1+restitution)*normal_scale*(-vn).clamp_min(0)
    for _ in range(iterations):
        for j in range(support):
            n,r=normals[:,j],arm[:,j]
            rel=linear+torch.cross(angular,r,dim=-1)-surface[:,j]
            old=normal_load[:,j]
            load=torch.where(active[:,j],(old+target[:,j]-(rel*n).sum(-1)).clamp_min(0),0.)
            linear=linear+(load-old)[:,None]*n
            normal_load[:,j]=load
            tangential=rel-(rel*n).sum(-1,keepdim=True)*n
            proposed=tangent_load[:,j]-(2/7)*tangential
            length=proposed.norm(dim=-1)
            bound=load*mu[:,j]
            value=proposed*(bound/length.clamp_min(1e-12)).clamp(max=1.)[:,None]
            diff=value-tangent_load[:,j]
            linear=linear+diff
            angular=angular+torch.cross(r,diff,dim=-1)/(.4*radius**2)
            tangent_load[:,j]=value
    return (normal_load[...,None]*normals).sum(1),tangent_load.sum(1)


@torch.no_grad()
def predict(trace,lanes,prior,material,settings,*,compound):
    t=trace['tensors'];g=prior.preview
    # Incoming states only. Broad phase includes every analytic racket primitive.
    source=t['shuttle_state_w'][:-1,lanes];p=source[...,:3]
    center=t['racket_position_w'][:-1,lanes];q=t['racket_quaternion_w'][:-1,lanes]
    result=torch.zeros_like(p);tangent=torch.zeros_like(p)
    count=torch.zeros(p.shape[:-1],dtype=torch.long)
    offset=settings['solver']['pair_contact_offset_m']
    bound=max((g.racket_box_centers.norm(dim=-1)+g.racket_box_halves.norm(dim=-1)).max().item(),
        (g.racket_round_centers.norm(dim=-1)+g.racket_round_halves.flatten()+g.racket_round_radii.flatten()).max().item())+g.radius+offset
    for index in ((p-center).norm(dim=-1)<=bound).nonzero().split(512):
        if not len(index):continue
        i,l=index.T
        pos=p[i,l];rot=q[i,l];origin=center[i,l]
        local=_quat_rotate_inverse(rot,pos-origin)
        args=(g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,g.racket_round_radii,g.racket_round_capsule)
        gaps=torch.cat([box_gap(local,g.racket_box_centers,g.racket_box_halves),round_gap(local,*args)],-1)-g.radius
        near_count=(gaps<=offset).sum(-1)
        nshape=int(near_count.max()) if compound else 1
        if not nshape:continue
        gap,ids=torch.topk(gaps,nshape,largest=False)
        axes=torch.cat([box_normal(local,g.racket_box_centers,g.racket_box_halves),round_normal(local,*args)],1)
        normal=_quat_rotate(rot[:,None],axes.gather(1,ids[...,None].expand(-1,-1,3)))
        arm=pos[:,None]-origin[:,None]-g.radius*normal
        omega=t['racket_angular_velocity_w'][i,torch.as_tensor(lanes)[l]]
        vc=t['racket_linear_velocity_w'][i,torch.as_tensor(lanes)[l]]
        surface=vc[:,None]+torch.cross(omega[:,None].expand_as(arm),arm,dim=-1)
        state=source[i,l]
        vn=((state[:,None,7:10]-surface)*normal).sum(-1)
        active=(gap<=offset)&(gap+prior.sub_dt*vn<=0)&(vn<0)
        nterm,tterm=solve_patch(normal,material[ids,1],material[ids,2],state[:,7:10],state[:,10:13],
             surface,g.radius,active,settings['fitted']['normal_impulse_scale'],4 if compound else 1)
        result[i,l]=nterm;tangent[i,l]=tterm;count[i,l]=active.sum(-1)
    return dict(normal=result,tangent=tangent,active_shapes=count)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('spec-dir','material','calibration','output'):
        cli.add_argument('--'+key,type=Path,required=True)
    a=cli.parse_args();torch.set_num_threads(3)
    runs={k:_load_branch(a.spec_dir,k) for k in BRANCHES}
    _same_contract(runs)
    status=json.loads((a.spec_dir.parent/'status.json').read_text())
    exp={r['branch']:r['probe_sha256'] for r in status['completed'] if r['spec']==a.spec_dir.name and r['exit_code']==0}
    hashes={k:sha(a.spec_dir/k/'native_probe.pt') for k in BRANCHES}
    if exp!=hashes:raise ValueError('Native branches not bound to capture')
    task=runs['nominal_a']['task_contract'];mat=validate_contract(json.loads(a.material.read_text()),task['asset']['sha256'])
    material=torch.tensor(mat['effective']); cfg=json.loads(a.calibration.read_text())
    prior=BallImpulsePriorV1().eval().requires_grad_(False)
    physics=json.loads((Path(__file__).resolve().parents[1]/'configs/shuttle_physics_v2.json').read_text())
    lanes=runs['nominal_a']['spec']['lanes'];groups={}
    for b,r in runs.items():
        trace=r['raw']['substep_trace'];t=trace['tensors'];dt=trace['physics_dt_s']
        if dt!=prior.sub_dt or trace['count']!=prior.steps+1:raise ValueError('Incompatible timebase')
        ref={k:t[field][1:,lanes]*(dt/physics['mass_kg']) for k,field in [('normal','racket_force_w'),('tangent','racket_friction_force_w')]}
        groups[b]=dict(native=ref,deepest=predict(trace,lanes,prior,material,cfg,compound=False),
            compound=predict(trace,lanes,prior,material,cfg,compound=True))
    samples=[]
    for j,lane in enumerate(lanes):
        row=dict(lane=lane,branch_events={b:mode(r,lane) for b,r in runs.items()})
        for model in ('native','deepest','compound'):
            record={}
            for key in ('normal','tangent'):
                data={b:v[model][key][:,j].sum(0) for b,v in groups.items()}
                torch.testing.assert_close(data['nominal_a'],data['nominal_b'],atol=1e-6,rtol=0)
                record[key]=(data['plus']-data['minus']).tolist()
            row[model]=record
        samples.append(row)
    aggregate={}
    for model in ('deepest','compound'):
        def vectors(name):
            return torch.tensor([r[name]['normal'] for r in samples])+torch.tensor([r[name]['tangent'] for r in samples])
        ref=vectors('native');pred=vectors(model)
        aggregate[model]=dict(component_rmse_mps=float((ref-pred).square().mean().sqrt()),
                gain_median=float((pred.norm(dim=-1)/ref.norm(dim=-1).clamp_min(1e-12)).median()),
                cosine_median=float(F.cosine_similarity(pred,ref).median()))
    out=dict(scope='Native incoming states supplied each substep, measured material per shape; differences are integrated racket impulses only',
             hashes=hashes,material_sha256=sha(a.material),calibration_sha256=sha(a.calibration),samples=samples,
             signed_impulse=aggregate,contact_bptt_admitted=False)
    with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
