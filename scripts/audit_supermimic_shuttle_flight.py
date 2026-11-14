#!/usr/bin/env python3
"""Compare pinned analytical free flight to recorded native transitions."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_shuttle_flight_v1 import ShuttleFreeFlightV1


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    manifest=json.loads((args.archive/'manifest.json').read_text())
    records=[]
    for entry in manifest['chunks']:
        path=args.archive/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Changed archive chunk')
        for row in torch.load(path,map_location='cpu',weights_only=True)['rows']:
            before,after=row['before'],row['after']
            b,a=before['task'],after['task']
            # This is selection by recorded events for offline diagnosis,
            # not a runtime controller using future labels. Absence of a
            # recorded event is not proof of zero native contact at substeps.
            mask=b['launched']&a['launched']&b['in_play']&a['in_play']
            mask &= before['physical']['ball_present']&after['physical']['ball_present']
            mask &= (b['contact_count']==a['contact_count'])
            mask &= ((~b['had_contact'])&(~a['had_contact'])) | (b['released']&a['released'])
            for key in ('ground_seen','shuttle_net_contact','body_ball_fault'):
                mask &= ~b[key]&~a[key]
            mask &= ~(row['terminated']|row['timeout'])
            records.append((before['physical']['ball_position_w'].clone(),
                            before['physical']['ball_velocity_w'].clone(),
                            after['physical']['ball_position_w'].clone(),
                            after['physical']['ball_velocity_w'].clone(),mask.clone()))
    if not records:
        raise ValueError('No committed transitions')
    p,v,pn,vn,mask=(torch.stack([r[i] for r in records]) for i in range(5))
    predictor=ShuttleFreeFlightV1(control_dt_s=manifest['control_dt_s'],
                                physics_dt_s=manifest['task_contract']['physics_dt_s'])
    outcomes={}
    for length in (1,8,16,64):
        if len(mask)<length:continue
        valid=mask.unfold(0,length,1).all(-1)
        starts,lanes=valid.nonzero(as_tuple=True)
        if not len(starts):
            outcomes[str(length)]={'count':0};continue
        predicted_p,predicted_v=p[starts,lanes],v[starts,lanes]
        for _ in range(length):
            predicted_p,predicted_v=predictor(predicted_p,predicted_v)
        pe=(predicted_p-pn[starts+length-1,lanes]).abs()
        ve=(predicted_v-vn[starts+length-1,lanes]).abs()
        dist=pe.norm(dim=-1)
        worst=int(dist.argmax())
        outcomes[str(length)]=dict(count=len(starts),duration_s=length*manifest['control_dt_s'],
             position_component_mae_m=float(pe.mean()),velocity_component_mae_m_s=float(ve.mean()),
             position_euclidean_p95_m=float(torch.quantile(dist,.95)),
             position_euclidean_max_m=float(dist.max()),
             worst_start_tick=int(starts[worst]),worst_lane=int(lanes[worst]))
    result=dict(scope='Recorded-event-free intervals only; not a contact model or policy score',
                chunks=len(manifest['chunks']),ticks=len(records),
                selected_single_transitions=int(mask.sum()),substep_contact_trace_available=False,
                outcomes=outcomes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
