#!/usr/bin/env python3
"""Inspect worst held-out ball rollouts without optimizer or native simulation."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,GeometrySupportedBallV1
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from scripts.fit_supermimic_ball_residual_v2 import verified
from scripts.fit_supermimic_physical_world import model_args


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);inputs=json.loads((a.run/'inputs.json').read_text());r=inputs['source_receipt']
    source=verified(r['physical_checkpoint'],r['physical_checkpoint_sha256'])
    c=torch.load(source,map_location='cpu',weights_only=True)
    base=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).cuda();base.load_state_dict(c['model'],strict=True)
    candidate=torch.load(a.run/'ball.pt',map_location='cpu',weights_only=True)
    ball=BallResidualV2().cuda();ball.load_state_dict(candidate['model'],strict=True)
    preview=BallGeometryPreviewV1().cuda()
    if preview.geometry_hashes!=candidate['geometry_hashes']:raise ValueError('Changed geometry')
    model=GeometrySupportedBallV1(ball,preview)
    info=json.loads((source.parent/'source.json').read_text());old=inputs['source_inputs']
    data=torch.load(verified(old['dataset'],info['receipt']['dataset_sha256']),map_location='cpu',weights_only=True)['tensors']
    data={k:v.cuda() for k,v in data.items()}
    trace=old['trace_receipt'];res=torch.load(verified(trace['residuals'],trace['residuals_sha256']),map_location='cpu',weights_only=True)
    masks={k:v.cuda() for k,v in res['masks'].items()}
    suite=c['source_manifest']['task_contract']['suite'];path=verified(suite['path'],suite['sha256'])
    cases=json.loads(path.read_text())['cases'];launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
    pairs=torch.tensor(old['split']['validation_selected'],device='cuda');t,l=pairs.T
    state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
    predictions=[];mode_probabilities=[]
    with torch.no_grad():
        for k in range(64):
            prob,_=model.components(ball_geometry_features(state,base.geometry));mode_probabilities.append(prob)
            state,feet,present=ball_transition(base,model,state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],feet,present,elapsed,launch_batch)
            elapsed=elapsed+.01;predictions.append(state[:,49:])
        predicted=torch.stack(predictions,1);prob=torch.stack(mode_probabilities,1)
        idx=t[:,None]+torch.arange(64,device='cuda')[None];truth=data['next_state'][idx,l[:,None],49:]
        error=(predicted[:,-1,:3]-truth[:,-1,:3]).norm(dim=-1)
        below=(predicted[:,:,2]<0).any(1);native_below=(truth[:,:,2]<0).any(1)
        records=[]
        for j in error.argsort(descending=True)[:20].tolist():
            events={name:torch.where(masks[name][idx[j],l[j]])[0].cpu().tolist() for name in ('new_racket','new_ground','new_net')}
            records.append(dict(window_index=j,lane=int(l[j]),start_tick=int(t[j]),start_elapsed=float(data['elapsed'][t[j],l[j]]),
                final_position_error_m=float(error[j]),predicted_final_pv=predicted[j,-1].cpu().tolist(),
                true_final_pv=truth[j,-1].cpu().tolist(),predicted_min_ball_z=float(predicted[j,:,2].min()),
                true_min_ball_z=float(truth[j,:,2].min()),events_at_control_offsets=events,
                mode_probability_sums=prob[j].sum(0).cpu().tolist()))
        result=dict(scope='Held-out window diagnosis; lanes identify evidence only, never controller inputs',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256((a.run/'ball.pt').read_bytes()).hexdigest(),optimizer_updates=0,
            windows=len(t),predicted_center_below_z0_windows=int(below.sum()),native_center_below_z0_windows=int(native_below.sum()),
            below_ground_groups={str(flag):dict(count=int((below==flag).sum()),
                final_position_mae_m=float((predicted[below==flag,-1,:3]-truth[below==flag,-1,:3]).abs().mean()) if (below==flag).any() else None)
                for flag in (False,True)},worst20=records)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='worst20'},indent=2))


if __name__=='__main__':main()
