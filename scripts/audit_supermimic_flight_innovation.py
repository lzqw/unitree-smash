#!/usr/bin/env python3
"""Locate the strongest observed departure from free flight in each evaluated native window."""
import argparse
import json
from pathlib import Path

import torch

from scripts.evaluate_supermimic_fresh_wm_panel import digest
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1
from quadsmash.rl.supermimic_shuttle_flight_v1 import ShuttleFreeFlightV1


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('score','capture','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(3)
    score=json.loads(a.score.read_text())
    source=a.capture/'native.pt'
    if digest(source)!=score['native_capture_sha256']:raise ValueError('Different native flight data')
    data=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    anchor=score['uniform_window_errors']['anchor']
    trial=score['uniform_window_errors']['candidate']
    if [(r['tick'],r['lane'],r['history']) for r in anchor]!=[(r['tick'],r['lane'],r['history']) for r in trial]:
        raise ValueError('Attribution windows differ')
    capture=json.loads((a.capture/'metrics.json').read_text())
    if capture['panel_sha256']!=score['native_panel_sha256'] or capture['data_sha256']!=score['native_capture_sha256']:
        raise ValueError('Score does not match the declared capture')
    policy=Path(list(capture['weights'])[1])
    if digest(policy)!=capture['weights'][str(policy)]:raise ValueError('Behavior checkpoint changed')
    ck=torch.load(policy,map_location='cpu',weights_only=False,mmap=True)
    geom=FullBodyGeometryV1(ck['metadata']['task_contract']['asset']['sha256'])
    preview=BallGeometryPreviewV1()
    horizon=max(score['horizons'])
    time=torch.tensor([r['tick'] for r in anchor]);lane=torch.tensor([r['lane'] for r in anchor])
    starts=time[None]+torch.arange(horizon)[:,None]
    old=data['state'][starts,lane];new=data['next_state'][starts,lane]
    predicted=ShuttleFreeFlightV1()(old[...,49:52],old[...,52:])[1]
    live=data['present'][starts,lane].bool()&data['next_present'][starts,lane].bool()
    innovation=(new[...,52:]-predicted).norm(dim=-1)
    magnitude,offset=torch.where(live,innovation,0.).max(0)
    columns=torch.arange(len(lane))
    first,last=(ball_geometry_features(x[offset,columns],geom) for x in (old,new))
    # Endpoints come from native states. Interpolation is only a geometry proximity diagnostic.
    frac=torch.linspace(0,1,9).reshape(1,-1,1)
    local=torch.lerp(first[:,None,6:9],last[:,None,6:9],frac)
    world=torch.lerp(first[:,None,:3],last[:,None,:3],frac)
    gaps=torch.stack((preview.racket_gap(local).amin(1),(world[...,2]-preview.radius).amin(1),
                      preview.net_gap(world).amin(1)),-1)
    distances,indices=gaps.sort(-1)
    by_group={};rows=[]
    for i,(before,after) in enumerate(zip(anchor,trial,strict=True)):
        if magnitude[i]<.1:
            label='under_0.1_mps'
        elif distances[i,0]>.03:
            label='distant'
        elif distances[i,1]-distances[i,0]<.001:
            label='ambiguous'
        else:
            label=('racket','ground','net')[indices[i,0]]
        row=dict(lane=before['lane'],start=before['tick'],history=before['history'],
                 peak_tick=int(time[i]+offset[i]),peak_mps=float(magnitude[i]),gaps_m=gaps[i].tolist(),
                 closest_geometry=label,anchor_squared_error=before['squared_error'],candidate_squared_error=after['squared_error'])
        rows.append(row)
        category=by_group.setdefault(row['history'],{}).setdefault(label,dict(count=0,anchor_squared_error=0.,candidate_squared_error=0.))
        category['count']+=1
        for k in ('anchor_squared_error','candidate_squared_error'):category[k]+=row[k]
    with a.output.open('x') as f:
        json.dump(dict(by_group=by_group,windows=rows,source_sha256=score['native_capture_sha256'],score_sha256=digest(a.score),
                       scope='Geometry near the strongest observed velocity innovation in a 64-tick window. This proxy uses recorded states; object IDs are not native force labels.'),
                  f,indent=2,allow_nan=False)


if __name__=='__main__':main()
