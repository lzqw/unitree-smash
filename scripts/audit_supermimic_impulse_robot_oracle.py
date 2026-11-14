#!/usr/bin/env python3
"""Hindsight full robot injection with autoregressive ball; diagnostic only."""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from scripts.fit_supermimic_ball_residual_v2 import verified
from scripts.fit_supermimic_physical_world import model_args, write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);begin=time.perf_counter();inputs=json.loads((a.run/'inputs.json').read_text())
    source=verified(inputs['physical_checkpoint'],inputs['physical_checkpoint_sha256'])
    c=torch.load(source,map_location='cpu',weights_only=True)
    base=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).cuda()
    base.load_state_dict(c['model'],strict=True);base.eval().requires_grad_(False)
    model=BallImpulsePriorV1().cuda()
    if model.preview.geometry_hashes!=inputs['geometry_hashes']:raise ValueError('Changed geometry')
    data=torch.load(verified(inputs['dataset'],inputs['dataset_sha256']),map_location='cpu',weights_only=True)['tensors']
    data={k:v.cuda() for k,v in data.items()}
    masks=torch.load(verified(inputs['residuals'],inputs['residuals_sha256']),map_location='cpu',weights_only=True)['masks']
    masks={k:v.cuda() for k,v in masks.items()}
    suite=c['source_manifest']['task_contract']['suite'];cases=json.loads(verified(suite['path'],suite['sha256']).read_text())['cases']
    launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
    pairs=torch.tensor(inputs['split']['validation_selected'],device='cuda');t,l=pairs.T
    initial,_,_,feet,present,elapsed,lb=model_args(data,pairs,launch)
    # Three ablations all use the same prior. Only hindsight diagnostic state
    # injection differs. No future robot data are provided to an actual policy.
    all_results={}
    for mode in ('autonomous','oracle_robot','oracle_robot_and_ball_until_first_contact'):
        state=initial.clone();active=present.clone();clock=elapsed.clone();foot=feet.clone()
        metrics={};ball_history=[];event_records=[];seen=torch.zeros(len(t),device='cuda',dtype=torch.bool)
        for k in range(64):
            true=data['state'][t+k,l]
            if mode!='autonomous':state=torch.cat((true[:,:49],state[:,49:]),-1)
            if mode=='oracle_robot_and_ball_until_first_contact':
                # Reset ball only BEFORE (including input of) first native impact;
                # never after it. This isolates pre-impact drift from response.
                state=torch.cat((state[:,:49],torch.where(seen[:,None],state[:,49:],true[:,49:])), -1)
            hit=masks['new_racket'][t+k,l]
            seen |= hit
            features=ball_geometry_features(state,base.geometry)
            true_features=ball_geometry_features(true,base.geometry)
            ball,info=model(state,base.geometry,active)
            prior,next_active=base.prior(state,active,clock,lb)
            ball=torch.where(active[:,None],ball,prior[:,49:])
            physical,logits,_=base(state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],foot,active,clock,lb)
            for j in torch.where(hit)[0].tolist():
                event_records.append(dict(window_index=j,lane=int(l[j]),start_tick=int(t[j]),offset=k,
                    predicted_local_ball=features[j,6:9].cpu().tolist(),true_local_ball=true_features[j,6:9].cpu().tolist(),
                    predicted_relative_velocity=features[j,9:12].cpu().tolist(),true_relative_velocity=true_features[j,9:12].cpu().tolist(),
                    predicted_normal=info['normal'][j].cpu().tolist(),predicted_part=int(info['racket_part'][j]),
                    selected_object=int(info['object_id'][j]),
                    next_ball_error_m=float((ball[j,:3]-data['next_state'][t[j]+k,l[j],49:52]).norm())))
            state=torch.cat((physical[:,:49],ball),-1);active=next_active;clock=clock+.01;foot=logits.sigmoid()
            ball_history.append(ball)
            if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite diagnostic rollout')
            if k+1 not in (1,8,16,64):continue
            error=(ball-data['next_state'][t+k,l,49:]).abs()
            idx=t[:,None]+torch.arange(k+1,device='cuda')[None]
            hit_group=masks['new_racket'][idx,l[:,None]].any(1)
            metrics[str(k+1)]=dict(ball_position_mae_m=float(error[:,:3].mean()),
                ball_velocity_mae_m_s=float(error[:,3:].mean()),
                ball_position_p95_euclidean_m=float(torch.quantile(error[:,:3].norm(dim=-1),.95)),
                first_contact_group={str(flag):dict(count=int((hit_group==flag).sum()),
                    ball_position_mae_m=float(error[hit_group==flag,:3].mean()) if (hit_group==flag).any() else None) for flag in (False,True)})
        predicted=torch.stack(ball_history,1);truth=data['next_state'][t+63,l,49:]
        error=(predicted[:,-1,:3]-truth[:,:3]).norm(dim=-1)
        all_results[mode]=dict(metrics=metrics,impact_records=event_records,
            below_ground_windows=int((predicted[:,:,2]<0).any(1).sum()),
            worst=[dict(window_index=j,lane=int(l[j]),start_tick=int(t[j]),error_m=float(error[j]),
                predicted=predicted[j,-1].cpu().tolist(),truth=truth[j].cpu().tolist()) for j in error.argsort(descending=True)[:10].tolist()])
        print(json.dumps(dict(mode=mode,metrics=metrics)),flush=True)
    baseline=json.loads((a.run/'rollout_validation.json').read_text())
    diff=max(abs(all_results['autonomous']['metrics'][s][k]-baseline[s][k]) for s in baseline for k in ('ball_position_mae_m','ball_velocity_mae_m_s','ball_position_p95_euclidean_m'))
    if diff>2e-6:raise ValueError('Autonomous baseline did not reproduce: '+str(diff))
    torch.cuda.synchronize()
    write_json(a.output,dict(scope='Hindsight diagnostic only; future injections never used for actor training',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_run=str(a.run),all_optimizer_updates=0,source_metric_max_error=diff,
        seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),results=all_results))


if __name__=='__main__':main()
