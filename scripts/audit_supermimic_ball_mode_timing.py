#!/usr/bin/env python3
"""Read-only timing diagnosis; future labels are used only for hindsight audit."""
import argparse,json,hashlib
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from scripts.fit_supermimic_ball_residual_v2 import verified


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    inputs=json.loads((a.run/'inputs.json').read_text());receipt=inputs['source_receipt']
    source=verified(receipt['checkpoint'],receipt['checkpoint_sha256'])
    info=json.loads((source.parent/'source.json').read_text())
    path=verified(inputs['dataset'],info['receipt']['dataset_sha256'])
    data=torch.load(path,weights_only=True,map_location='cpu')['tensors']
    tr=inputs['trace_receipt'];res=torch.load(verified(tr['residuals'],tr['residuals_sha256']),weights_only=True,map_location='cpu')
    c=torch.load(source,weights_only=True,map_location='cpu')
    geometry=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).geometry.cuda()
    model=BallResidualV2().cuda();checkpoint=torch.load(a.run/'ball.pt',map_location='cpu',weights_only=True)
    if checkpoint['schema']!=model.schema:raise ValueError('Wrong candidate')
    model.load_state_dict(checkpoint['model'],strict=True)
    lanes=torch.tensor(inputs['split']['validation_lanes'])
    times=torch.arange(len(data['state']))[:,None].expand(-1,len(lanes)).flatten()
    lane=lanes[None,:].expand(len(data['state']),-1).flatten()
    with torch.no_grad():
        f=torch.cat([ball_geometry_features(s.cuda(),geometry).cpu() for s in data['state'][times,lane].split(2048)])
        probs=torch.cat([model.components(x.cuda())[0].cpu() for x in f.split(2048)])
    output={'scope':'Hindsight labels for timing audit only; no optimization or runtime label access',
        'checkpoint_sha256':hashlib.sha256((a.run/'ball.pt').read_bytes()).hexdigest(),'groups':{}}
    free=res['masks']['recorded_event_free'][times,lane]
    for mode,event in ((1,'new_racket'),(2,'new_ground'),(3,'new_net')):
        selected=free & (probs.argmax(-1)==mode)
        t,l=times[selected],lane[selected];n=len(t)
        nearest=torch.full((n,),999,dtype=torch.long)
        for offset in sorted(range(-16,17),key=lambda x:(abs(x),x)):
            j=t+offset;valid=(j>=0)&(j<len(data['state']));safe=j.clamp(0,len(data['state'])-1)
            hit=valid & res['masks'][event][safe,l] & (data['episode'][safe,l]==data['episode'][t,l])
            nearest=torch.where(hit & (nearest==999),offset,nearest)
        values,counts=nearest.unique(return_counts=True)
        fs=f[selected]
        def quant(x):return [float(torch.quantile(x,q)) for q in (.1,.5,.9)] if n else None
        output['groups'][event]={'false_positive_recorded_free_count':n,
            'nearest_first_event_offset_ticks':{str(int(k)):int(v) for k,v in zip(values,counts)},
            'positive_offset_means_future_event':True,'999_means_no_recorded_first_event_within_16_ticks':True,
            'abs_racket_normal_distance_m_p10_p50_p90':quant(fs[:,6].abs()),
            'racket_radial_distance_m_p10_p50_p90':quant(fs[:,7:9].norm(dim=-1)),
            'ball_ground_clearance_m_p10_p50_p90':quant(fs[:,21]),
            'ball_vertical_speed_m_s_p10_p50_p90':quant(fs[:,5])}
    a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
