#!/usr/bin/env python3
"""Extract native task traces and audit free-flight residuals; no new simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1


FIELDS=('launched','in_play','had_contact','string_hit','frame_hit','released','ground_seen',
        'shuttle_net_contact','body_ball_fault','contact_count','quiet_frames')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path);parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    manifest=json.loads((args.archive/'manifest.json').read_text());before=[];after=[]
    for entry in manifest['chunks']:
        path=args.archive/entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Changed native chunk')
        rows=torch.load(path,map_location='cpu',weights_only=True)['rows']
        for row in rows:
            for dest,side in ((before,'before'),(after,'after')):
                dest.append({k:row[side]['task'][k].clone() for k in FIELDS})
    b={k:torch.stack([x[k] for x in before]) for k in FIELDS}
    a={k:torch.stack([x[k] for x in after]) for k in FIELDS}
    d=torch.load(args.dataset,map_location='cpu',weights_only=True)['tensors']
    if b['launched'].shape!=d['episode'].shape:raise ValueError('Trace/data dimensions differ')
    torch.save(dict(before=b,after=a,source_manifest=manifest),args.output/'ball_trace.pt')
    suite=manifest['task_contract']['suite'];cases=json.loads(Path(suite['path']).read_text())['cases']
    if hashlib.sha256(Path(suite['path']).read_bytes()).hexdigest()!=suite['sha256']:raise ValueError('Changed launch suite')
    device='cuda:0';model=PhysicalWorldV1(hidden=(16,)).to(device)
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],device=device)
    priors=[]
    with torch.no_grad():
        for start in range(0,len(d['state']),16):
            state=d['state'][start:start+16].to(device);shape=state.shape[:2]
            prior,_=model.prior(state.flatten(0,1),d['ball_present'][start:start+16].flatten().to(device),
                d['elapsed'][start:start+16].flatten().to(device),launch.repeat(shape[0],1))
            priors.append(prior[:,49:].reshape(*shape,6).cpu())
    prior=torch.cat(priors);delta=d['next_state'][:,:,49:]-prior
    masks=dict(parked=~b['launched'],launch=(~b['launched'])&a['launched'],
        new_racket=a['contact_count']>b['contact_count'],
        new_ground=(~b['ground_seen'])&a['ground_seen'],
        new_net=(~b['shuttle_net_contact'])&a['shuttle_net_contact'],
        new_body=(~b['body_ball_fault'])&a['body_ball_fault'],
        after_ground=b['ground_seen'],after_net=b['shuttle_net_contact'],
        in_play=b['in_play'],after_release=b['released'])
    free=b['launched']&a['launched']&b['in_play']&a['in_play']
    free &= (a['contact_count']==b['contact_count']) & ((~b['had_contact']&~a['had_contact'])|(b['released']&a['released']))
    for name in ('ground_seen','shuttle_net_contact','body_ball_fault'):free &= ~b[name]&~a[name]
    masks['recorded_event_free']=free
    masks['not_recorded_event_free']=b['launched']&~free
    stats={}
    for name,mask in masks.items():
        values=delta[mask];n=len(values)
        stats[name]=dict(count=n,position_component_mae_m=float(values[:,:3].abs().mean()) if n else None,
            velocity_component_mae_m_s=float(values[:,3:].abs().mean()) if n else None,
            velocity_residual_p95_m_s=float(torch.quantile(values[:,3:].norm(dim=-1),.95)) if n else None,
            velocity_residual_max_m_s=float(values[:,3:].norm(dim=-1).max()) if n else None)
    torch.save(dict(ball_prior=prior,ball_residual=delta,masks=masks),args.output/'ball_residuals.pt')
    result=dict(scope='Overlapping recorded-event groups; not complete substep contact labels',
        source_archive=str(args.archive),dataset=str(args.dataset),statistics=stats)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
