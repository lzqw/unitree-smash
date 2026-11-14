#!/usr/bin/env python3
"""Hindsight robot-state injection to locate ball error, never deployable inference."""
import argparse,json,hashlib,subprocess,time
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,GeometrySupportedBallV1
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from scripts.fit_supermimic_ball_residual_v2 import verified
from scripts.fit_supermimic_physical_world import model_args


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);start=time.perf_counter()
    settings=json.loads((a.run/'inputs.json').read_text());inputs=settings['source_inputs']
    src=settings['physical_receipt'];physicalpath=verified(src['checkpoint'],src['checkpoint_sha256'])
    physical=torch.load(physicalpath,map_location='cpu',weights_only=True)
    base=PhysicalWorldV2(physical['source_manifest']['task_contract']['asset']['sha256']).cuda()
    base.load_state_dict(physical['model'],strict=True)
    receipt=settings['ball_receipt'];ballpath=verified(receipt['checkpoint'],receipt['checkpoint_sha256'])
    ball=BallResidualV2().cuda();ball.load_state_dict(torch.load(ballpath,map_location='cpu',weights_only=True)['model'],strict=True)
    preview=BallGeometryPreviewV1().cuda()
    if preview.geometry_hashes!=settings['geometry_hashes']:raise ValueError('Geometry changed')
    model=GeometrySupportedBallV1(ball,preview)
    info=json.loads((physicalpath.parent/'source.json').read_text())
    data=torch.load(verified(inputs['dataset'],info['receipt']['dataset_sha256']),map_location='cpu',weights_only=True)['tensors']
    data={k:v.cuda() for k,v in data.items()}
    suite=physical['source_manifest']['task_contract']['suite'];path=verified(suite['path'],suite['sha256'])
    cases=json.loads(path.read_text())['cases']
    launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
    pairs=torch.tensor(inputs['split']['validation_selected'],device='cuda');t,l=pairs.T
    state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
    metrics={}
    with torch.no_grad():
        for k in range(64):
            # Deliberate oracle injection: future recorded robot state only.
            # Predicted ball remains autoregressive. Never used for actor learning.
            state=torch.cat((data['state'][t+k,l,:49],state[:,49:]),-1)
            prior,active=base.prior(state,present,elapsed,launch_batch)
            correction,_=model(ball_geometry_features(state,base.geometry),present)
            state=torch.cat((state[:,:49],prior[:,49:]+correction),-1)
            present=active;elapsed=elapsed+.01
            if k+1 not in (1,8,16,64):continue
            error=(state[:,49:]-data['next_state'][t+k,l,49:]).abs()
            idx=t[:,None]+torch.arange(k+1,device='cuda')[None]
            events=data['next_event_state'][idx,l[:,None]]>data['event_state'][idx,l[:,None]]
            hit=events[:,:,[1,6]].any(-1).any(-1)
            metrics[str(k+1)]=dict(ball_position_mae_m=float(error[:,:3].mean()),
                ball_velocity_mae_m_s=float(error[:,3:].mean()),
                ball_position_p95_euclidean_m=float(torch.quantile(error[:,:3].norm(dim=-1),.95)),
                first_contact_group={str(flag):{'count':int((hit==flag).sum()),
                    'ball_position_mae_m':float(error[hit==flag,:3].mean())} for flag in (False,True)})
    result=dict(scope='Oracle future robot state, diagnostic only; no autonomous or task success claim',
        original_run=str(a.run),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),all_optimizer_updates=0,
        metrics=metrics,seconds=time.perf_counter()-start)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
