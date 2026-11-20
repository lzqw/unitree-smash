#!/usr/bin/env python3
"""Compare one- and multi-shape contact solves with incoming measured shuttle and racket states."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_supermimic_solver_action_response import BallImpulsePriorV1, BRANCHES, mode, sha
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_compound_racket import compound_step, solve_patch
from quadsmash.rl.supermimic_racket_material_contract_v1 import validate_contract


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
        state=source[i,l]
        source_lane=torch.as_tensor(lanes,device=l.device)[l]
        result[i,l],tangent[i,l],_,count[i,l]=compound_step(
            prior,p[i,l],state[:,7:10],state[:,10:13],center[i,l],q[i,l],
            t['racket_linear_velocity_w'][i,source_lane],t['racket_angular_velocity_w'][i,source_lane],
            material,offset,settings['fitted']['normal_impulse_scale'],compound=compound)
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
