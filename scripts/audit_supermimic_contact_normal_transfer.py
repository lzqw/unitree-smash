#!/usr/bin/env python3
"""Attribute signed impulse error to collision timing, primitive normals and measured native contact geometry."""
import argparse
import json
from pathlib import Path
import torch

from audit_supermimic_paired_body_transfer import response_metrics
from audit_supermimic_surface_derivative_scales import read_material
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,sha
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap,round_gap
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import box_normal,round_normal
from quadsmash.rl.supermimic_compound_racket import solve_patch
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate,_quat_rotate_inverse


def replay_geometry(trace,prior,material,cfg,*,separate_impulses=False):
    t=trace['tensors'];shuttle=t['shuttle_state_w']
    names=('analytic_at_loaded_times','native_normal','native_normal_and_point','native_normal_point_separation')
    steps,lanes=shuttle.shape[:2]
    outputs={k:shuttle.new_zeros(lanes,3) for k in names}
    inputs=[];normals=[];points=[];gaps=[]
    for tick,frame in enumerate(trace['manifold_frames'][1:]):
        use=(frame['filter_index']==0)&(frame['normal_force_n'].abs()>.001)
        if use.any():
            lane=frame['lane'][use].long()
            inputs.append(tick*lanes+lane)
            normals.append(frame['normal_w'][use]);points.append(frame['point_w'][use]);gaps.append(frame['separation_m'][use])
    if separate_impulses:
        for name in names:
            for channel in ('normal','tangent'):
                outputs[name+'_'+channel]=shuttle.new_zeros(lanes,3)
    if not inputs:return outputs,{}
    flat=torch.cat(inputs)
    order=torch.argsort(flat,stable=True);flat=flat[order]
    ids,inv,counts=torch.unique_consecutive(flat,return_counts=True,return_inverse=True)
    slot=torch.arange(len(flat))-(counts.cumsum(0)-counts)[inv]
    T,L=ids//lanes,ids%lanes
    state=shuttle[T,L];p=state[:,:3];v=state[:,7:10];w=state[:,10:13]
    c,q,linear,omega=(t[k][T,L] for k in ('racket_position_w','racket_quaternion_w','racket_linear_velocity_w','racket_angular_velocity_w'))
    m=int(counts.max());active=torch.arange(m)[None,:]<counts[:,None]
    observed=state.new_zeros(len(ids),m,3);point=observed.clone();reported=state.new_zeros(len(ids),m)
    observed[inv,slot]=torch.cat(normals)[order];point[inv,slot]=torch.cat(points)[order]
    reported[inv,slot]=torch.cat(gaps)[order]
    # Orient a native manifold's axis toward the incoming shuttle center.
    observed*=torch.where(((p[:,None]-point)*observed).sum(-1,keepdim=True)<0,-1.,1.)
    l=_quat_rotate_inverse(q,p-c);g=prior.preview
    rounded=(g.racket_round_centers,g.racket_round_axes,g.racket_round_halves,g.racket_round_radii,g.racket_round_capsule)
    distances=torch.cat((box_gap(l,g.racket_box_centers,g.racket_box_halves),round_gap(l,*rounded)),-1)-g.radius
    axes=torch.cat((box_normal(l,g.racket_box_centers,g.racket_box_halves),round_normal(l,*rounded)),1)
    # Bind each reported contact to a primitive by its reported location, independent of force magnitude.
    lp=_quat_rotate_inverse(q[:,None],point-c[:,None])
    point_distance=torch.cat((box_gap(lp,g.racket_box_centers,g.racket_box_halves),round_gap(lp,*rounded)),-1).abs()
    sid=point_distance.argmin(-1)
    axis=_quat_rotate(q[:,None],axes.gather(1,sid[...,None].expand(-1,-1,3)))
    gap=distances.gather(1,sid)
    settings=cfg['fitted']
    for key in names:
        n=axis if key==names[0] else observed
        r=p[:,None]-g.radius*n if key in names[:2] else point
        distance=reported if key==names[-1] else gap
        surface=linear[:,None]+torch.linalg.cross(omega[:,None].expand_as(r),r-c[:,None])
        normal,tangent=solve_patch(n,material[sid,1],material[sid,2],v,w,surface,g.radius,active,
                    settings['normal_impulse_scale'],4,separation_velocity=settings.get('separation_velocity_gain',0.)*distance/prior.sub_dt)
        outputs[key].index_add_(0,L,normal+tangent)
        if separate_impulses:
            outputs[key+'_normal'].index_add_(0,L,normal)
            outputs[key+'_tangent'].index_add_(0,L,tangent)
    return outputs,dict(normal_cosine=(axis*observed).sum(-1)[active],gap_error=(gap-reported)[active],
                        radial_point_error=((p[:,None]-g.radius*observed)-point).norm(dim=-1)[active])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('wide','material','readout','calibration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();torch.set_num_threads(2)
    cfg=json.loads(a.calibration.read_text())
    prov=json.loads((a.wide/'nominal_a/provenance.json').read_text())
    material=read_material(a.wide,prov,a.material,a.readout,a.calibration)
    prior=BallImpulsePriorV1()
    reports={}
    for split in ('train','development'):
        audit=json.loads((a.wide/(split+'.json')).read_text())
        rows=sorted(audit['samples'],key=lambda r:(r['seed'],r['lane']))
        sums={};points={}
        for seed in sorted({r['seed'] for r in rows}):
            trials={}
            for b in ('minus','plus'):
                path=a.wide/b/f'spec_{seed}'/'trace.pt'
                if sha(path)!=audit['archives'][str(path)]:raise ValueError('Changed native signed trajectories')
                trace=torch.load(path,mmap=True,weights_only=True)['raw']['substep_trace']
                trials[b],angles=replay_geometry(trace,prior,material,cfg)
                for k,v in angles.items():points.setdefault(k,[]).append(v)
            for k,v in trials['plus'].items():sums.setdefault(k,[]).append(v-trials['minus'][k])
        native=torch.tensor([r['delta']['native'] for r in rows]);repeat=torch.tensor([r['repeat']['native'] for r in rows])
        stats={}
        for label,mask in [('all',torch.ones(len(rows),dtype=torch.bool)),('stable',torch.tensor([r['kind']=='racket_stable' for r in rows]))]:
            stats[label]={k:response_metrics(native[mask],torch.cat(v)[mask],repeat[mask]) for k,v in sums.items()}
        stats['point_distribution']={k:torch.cat(v).quantile(torch.tensor([.1,.5,.9])).tolist() for k,v in points.items()}
        reports[split]=stats
    with a.output.open('x') as f:
        json.dump(dict(groups=reports,scope='Uses observed loaded contact times and count; discrete shape assigned by distance to reported point. Force magnitude excluded from solve.',
            fit_sha256=sha(a.readout),calibration_sha256=sha(a.calibration),policy_updates=0,contact_gradients_admitted=False),f,indent=2,allow_nan=False)


if __name__=='__main__':main()
