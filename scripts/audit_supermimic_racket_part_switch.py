"""Identify first predicted racket components on signed native action paths, including near ties."""
import argparse
import json
from pathlib import Path

import torch

from scripts.evaluate_supermimic_contact_rollout import sha
from scripts.fit_supermimic_native_racket_toi_head import g07, prepare, predicted_state
from scripts.calibrate_supermimic_toi_slip import TOISlip
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap, round_gap
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1, box_normal, round_normal
from quadsmash.rl.supermimic_racket_material_contract_v1 import component_names


def nearest_parts(prior, local, selected):
    geometry=prior.preview
    box=(geometry.racket_box_centers,geometry.racket_box_halves)
    rounded=(geometry.racket_round_centers,geometry.racket_round_axes,geometry.racket_round_halves,
             geometry.racket_round_radii,geometry.racket_round_capsule)
    gaps=torch.cat((box_gap(local,*box),round_gap(local,*rounded)),-1)
    normals=torch.cat((box_normal(local,*box),round_normal(local,*rounded)),-2)
    ids=torch.arange(len(local))
    first=gaps[ids,selected]
    remaining=gaps.clone()
    remaining[ids,selected]=torch.inf
    second,other=remaining.min(-1)
    if (second < first-1.e-6).any():
        raise ValueError('Contact reconstruction chose a different part')
    return other,second-first,(normals[ids,selected]*normals[ids,other]).sum(-1)


@torch.no_grad()
def inspect(root,prior,model):
    _,_,_,groups,sequences,identity=g07(root,prior)
    names=component_names()
    panel={}
    for branch,sequence in sequences.items():
        state=sequence[0][0]
        records={}
        for tick,(_,target,stream) in enumerate(sequence):
            row=prepare(prior,state,target,stream)
            active=row['active'] & (row['object_id']==0)
            for j in active.nonzero().flatten().tolist():
                lane=identity['lanes'][j]
                if lane in records:
                    continue
                part=row['racket_part'][j:j+1]
                other,margin,cosine=nearest_parts(prior,row['local'][j:j+1],part)
                n=row['normal'][j]
                velocity=row['relative_velocity'][j]
                records[lane]=dict(tick=tick,time_s=tick*prior.dt+float(row['time'][j]),group=groups[j],
                    part=names[int(part)],runner_up=names[int(other)],gap_margin_m=float(margin),
                    normal_cosine=float(cosine),sticking_mps=model.sticking*float((velocity-n*n.dot(velocity)).norm()),
                    normal_cap_mps=model.friction*float(row['impulse_dv'][j].norm()))
            state=predicted_state(model,prior,row)
        panel[branch]=records
    return dict(status_sha256=sha(root/'status.json'),branches=panel)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('calibration','fit','seam-fit','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--training',action='append',type=Path,required=True)
    parser.add_argument('--development',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    fit,seams=read(args.fit),read(args.seam_fit)
    if (seams['fit_sha256']!=sha(args.fit) or fit['calibration_sha256']!=sha(args.calibration)
            or seams['mechanism']!='strip_seam_sticking'):
        raise ValueError('Expected the same analytic calibration and completed seam comparison')
    prior=BallImpulsePriorV1(normal_impulse_scale=read(args.calibration)['fitted']['normal_impulse_scale']).eval()
    model=TOISlip(fit['selected']['friction'],fit['selected']['sticking'])
    panels={}
    for directory in args.training:
        receipt=directory/'support.json'
        if (sha(receipt) not in seams['additional_support'].values() or not read(receipt)['training_only']
                or sha(directory/'raw/status.json')!=read(receipt)['status_sha256']):
            raise ValueError('Unbound training intervention')
        panels[directory.name]=inspect(directory/'raw',prior,model)
    dev=read(args.development/'completed.json')
    if fit['g07_sha256']!=dev['wide']:
        raise ValueError('Unbound development cases')
    for scale in ('wide','half','quarter'):
        directory=args.development/scale
        if sha(directory/'status.json')!=dev[scale]:
            raise ValueError('Changed native development scale')
        panels[scale]=inspect(directory,prior,model)
    with args.output.open('x') as stream:
        json.dump(dict(panels=panels,fit_sha256=sha(args.fit),seam_fit_sha256=sha(args.seam_fit),
                       scope='Racket part names and geometry margins are predictions at the first analytic TOI. Native group labels use measured contact loads.'),
                  stream,indent=2,allow_nan=False)
