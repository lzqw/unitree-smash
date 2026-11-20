#!/usr/bin/env python3
"""Audit whether equal PhysX contact point counts represent the same racket primitives."""
import argparse
import json
from pathlib import Path
import torch

from audit_supermimic_native_window_response import complete_window
from audit_supermimic_paired_body_transfer import response_metrics
from evaluate_supermimic_contact_rollout import BallImpulsePriorV1,sha
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import box_gap,round_gap
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch,_same_contract
from quadsmash.rl.supermimic_contact_pilot_v1 import _quat_rotate_inverse


def infer_shape(distance, uncertainty):
    nearest,ids=distance.abs().topk(2,dim=-1,largest=False)
    # Both candidate surfaces have 1-Lipschitz point distance. Retain unresolved points.
    unambiguous=(nearest[:,1]-nearest[:,0])>2*uncertainty
    return torch.where(unambiguous,ids[:,0],-1)


def shape_history(traces, geometry):
    lanes=traces[0]['tensors']['shuttle_state_w'].shape[1]
    labels=[[] for _ in range(lanes)]
    rounded=(geometry.racket_round_centers,geometry.racket_round_axes,geometry.racket_round_halves,
             geometry.racket_round_radii,geometry.racket_round_capsule)
    start=0
    for trace in traces:
        t=trace['tensors']
        for i,frame in enumerate(trace['manifold_frames'][1:]):
            keep=(frame['filter_index']==0)&(frame['normal_force_n'].abs()>.001)
            if not keep.any():
                continue
            lane=frame['lane'][keep].long()
            point=frame['point_w'][keep]
            center=t['racket_position_w'][i,lane];q=t['racket_quaternion_w'][i,lane]
            local=_quat_rotate_inverse(q,point-center)
            dist=torch.cat((box_gap(local,geometry.racket_box_centers,geometry.racket_box_halves),
                            round_gap(local,*rounded)),-1)
            roundoff=4*torch.finfo(point.dtype).eps*torch.maximum(point.abs(),center.abs()).amax(-1).clamp_min(1)
            # The recorder exports contact position and signed pair separation, without primitive IDs.
            # Allow the entire separation as well as coordinate subtraction roundoff.
            ids=infer_shape(dist,frame['separation_m'][keep].abs()+roundoff)
            for j in lane.unique().tolist():
                labels[j].append((start+i,tuple(sorted(ids[lane==j].tolist()))))
        start+=len(trace['manifold_frames'])-1
    return labels


def shape_class(histories):
    if any(not h or any(-1 in ids for _,ids in h) for h in histories):
        return 'unresolved'
    return 'consistent' if all(h==histories[0] for h in histories[1:]) else 'changed'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('collection','response','local-gradient','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    read=lambda path:json.loads(path.read_text())
    result=read(a.response);local=read(a.local_gradient);status=read(a.collection/'status.json')
    if (status['status']!='COLLECTION_FINISHED' or result['status_sha256']!=sha(a.collection/'status.json')
            or local['source_sha256']!=sha(a.response)):
        raise ValueError('Native command window or local gradient identity mismatch')
    prior=BallImpulsePriorV1().eval()
    groups={};runs={}
    for r in status['completed']:
        b=r['branch'];d=a.collection/r['spec']
        if sha(d/b/'native_probe.pt')!=r['probe_sha256']:raise ValueError('Changed physical branch')
        runs[b]=_load_branch(d,b)
    _same_contract(runs)
    for b,run in runs.items():
        groups[b]=shape_history(complete_window(run['raw'],run['spec'],prior.steps),prior.preview)
    cases=result['cases']
    if local['lanes']!=[r['lane'] for r in cases] or list(runs['plus']['spec']['lanes'])!=local['lanes']:
        raise ValueError('Misaligned native and model cases')
    rows=[]
    for i,r in enumerate(cases):
        signatures={b:v[i] for b,v in groups.items()}
        rows.append(dict(lane=r['lane'],count_based=r['group'],shape_assignment=shape_class(list(signatures.values())),
                         native_point_attribution=signatures))
    final=str(result['ticks'][-1])
    native,repeat=(torch.tensor([r['ticks'][final][name] for r in cases]) for name in ('native','native_repeat'))
    stats={}
    for name in ('all','count_stable','consistent','changed','unresolved'):
        mask=torch.tensor([name=='all' or (r['count_based']=='racket_stable' and
                           (name=='count_stable' or name==r['shape_assignment'])) for r in rows])
        if mask.any():
            stats[name]={k:response_metrics(native[mask],torch.tensor(v)[mask],repeat[mask]) for k,v in local['local_response'].items()}
    with a.output.open('x') as f:
        json.dump(dict(groups=stats,cases=rows,source_sha256=sha(a.response),local_gradient_sha256=sha(a.local_gradient),
                  scope='Post-hoc nearest primitive inferred from reported points; ambiguous matches retained',
                  contact_gradients_admitted=False),f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':main()
