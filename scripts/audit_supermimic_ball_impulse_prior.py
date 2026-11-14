#!/usr/bin/env python3
"""Read-only GPU check of a single-impact physical prior against native data.

No native simulation, policy learning, fitting, or future contact labels in
prediction. Recorded controls are used only as open-loop dynamics inputs.
"""
import argparse, hashlib, json, subprocess, time, traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_impulse_prior_v1 import BallImpulsePriorV1
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from scripts.fit_supermimic_ball_residual_v2 import verified
from scripts.fit_supermimic_physical_world import model_args, write_json


def group_stats(predicted, truth, info, masks):
    out={}
    for name, mask in masks.items():
        count=int(mask.sum())
        if not count:
            out[name]={'count':0};continue
        error=predicted[mask]-truth[mask]
        out[name]=dict(count=count,position_mae_m=float(error[:,:3].abs().mean()),
            velocity_mae_m_s=float(error[:,3:].abs().mean()),
            velocity_direction_cosine=float(torch.nn.functional.cosine_similarity(predicted[mask,3:],truth[mask,3:]).mean()),
            detected_objects=torch.bincount(info['object_id'][mask],minlength=4).cpu().tolist(),
            normal_speed_mean=float(info['normal_speed'][mask].mean()),
            impulse_velocity_mean=float(info['impulse_dv'][mask].norm(dim=-1).mean()))
    return out


@torch.no_grad()
def run(a):
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(2);begin=time.perf_counter()
    inputs=json.loads((a.source_run/'inputs.json').read_text());r=inputs['source_receipt'];old=inputs['source_inputs']
    physicalpath=verified(r['physical_checkpoint'],r['physical_checkpoint_sha256'])
    checkpoint=torch.load(physicalpath,map_location='cpu',weights_only=True)
    base=PhysicalWorldV2(checkpoint['source_manifest']['task_contract']['asset']['sha256']).cuda()
    base.load_state_dict(checkpoint['model'],strict=True);base.eval().requires_grad_(False)
    model=BallImpulsePriorV1().cuda().eval()
    if model.preview.geometry_hashes!=r['geometry_hashes']:raise ValueError('Changed original geometry')
    metadata=json.loads((physicalpath.parent/'source.json').read_text())
    dataset=verified(old['dataset'],metadata['receipt']['dataset_sha256'])
    raw=torch.load(dataset,map_location='cpu',weights_only=True)['tensors']
    data={k:v.cuda() for k,v in raw.items()};del raw
    trace=old['trace_receipt']
    residualpath=verified(trace['residuals'],trace['residuals_sha256'])
    res=torch.load(residualpath,map_location='cpu',weights_only=True)
    masks={k:v.cuda() for k,v in res['masks'].items()};del res
    suite=checkpoint['source_manifest']['task_contract']['suite']
    cases=json.loads(verified(suite['path'],suite['sha256']).read_text())['cases']
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],device='cuda')
    write_json(a.output/'inputs.json',dict(physical_checkpoint=str(physicalpath),physical_checkpoint_sha256=r['physical_checkpoint_sha256'],
        dataset=str(dataset),dataset_sha256=metadata['receipt']['dataset_sha256'],residuals=str(residualpath),
        residuals_sha256=trace['residuals_sha256'],geometry_hashes=model.preview.geometry_hashes,
        model_schema=model.schema,restitution=model.restitution,split=old['split'],
        limitations=['single impact','kinematic infinite-mass surface','no friction/spin/effective mass',
                     'approximate swept path','no contact persistence or penetration correction',
                     'discrete selection is not a smooth actor objective']))
    one_step={};examples=[]
    all_pairs=torch.stack(torch.meshgrid(torch.arange(len(data['state']),device='cuda'),
                                        torch.arange(data['state'].shape[1],device='cuda'),indexing='ij'),-1)
    for split in ('training','validation'):
        lanes=torch.tensor(old['split'][split+'_lanes'],device='cuda')
        pairs=all_pairs[:,lanes].reshape(-1,2)
        if split=='training':
            t,l=pairs.T;pairs=pairs[(masks['new_racket']|masks['new_ground']|masks['new_net'])[t,l]]
        t,l=pairs.T;outputs=[];infos=[]
        for batch in pairs.split(256):
            state,_,_,feet,present,elapsed,lb=model_args(data,batch,launch)
            predicted,info=model(state,base.geometry,present)
            prior,_=base.prior(state,present,elapsed,lb)
            predicted=torch.where(present[:,None],predicted,prior[:,49:])
            if not torch.isfinite(predicted).all():raise FloatingPointError('Nonfinite physical prior')
            outputs.append(predicted);infos.append(info)
        pred=torch.cat(outputs);info={k:torch.cat([row[k] for row in infos]) for k in infos[0]}
        names=('new_racket','new_ground','new_net','recorded_event_free','after_ground')
        selected={name:masks[name][t,l] for name in names if name in masks}
        selected['all_selected']=torch.ones(len(t),device='cuda',dtype=torch.bool)
        truth=data['next_state'][t,l,49:]
        one_step[split]=group_stats(pred,truth,info,selected)
        hit=selected['new_racket'];error=(pred[:,3:]-truth[:,3:]).norm(dim=-1)
        ranking=torch.where(hit,error,torch.full_like(error,-1.)).argsort(descending=True)[:min(12,int(hit.sum()))]
        for j in ranking.tolist():
            examples.append(dict(split=split,lane=int(l[j]),tick=int(t[j]),
                current_ball=data['state'][t[j],l[j],49:].cpu().tolist(),predicted=pred[j].cpu().tolist(),truth=truth[j].cpu().tolist(),
                **{k:v[j].cpu().tolist() for k,v in info.items()}))
        print(json.dumps(dict(stage='one_step_finished',split=split,groups=one_step[split])),flush=True)
    write_json(a.output/'one_step.json',one_step);write_json(a.output/'worst_impacts.json',examples)
    pairs=torch.tensor(old['split']['validation_selected'],device='cuda');t,l=pairs.T
    state,_,_,feet,present,elapsed,lb=model_args(data,pairs,launch)
    result={};history=[];objects=[]
    for k in range(64):
        ball,info=model(state,base.geometry,present)
        prior,_=base.prior(state,present,elapsed,lb)
        ball=torch.where(present[:,None],ball,prior[:,49:])
        physical,logits,active=base(state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],feet,present,elapsed,lb)
        state=torch.cat((physical[:,:49],ball),-1);feet=logits.sigmoid();present=active;elapsed=elapsed+.01
        if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite autoregressive state')
        history.append(ball);objects.append(info['object_id'])
        if k+1 not in (1,8,16,64):continue
        truth=data['next_state'][t+k,l];error=(state-truth).abs()
        idx=t[:,None]+torch.arange(k+1,device='cuda')[None]
        first=masks['new_racket'][idx,l[:,None]].any(1)
        result[str(k+1)]=dict(ball_position_mae_m=float(error[:,49:52].mean()),
            ball_position_p95_euclidean_m=float(torch.quantile((state[:,49:52]-truth[:,49:52]).norm(dim=-1),.95)),
            ball_velocity_mae_m_s=float(error[:,52:].mean()),joint_mae_rad=float(error[:,13:31].mean()),
            first_contact_group={str(flag):dict(count=int((first==flag).sum()),
                ball_position_mae_m=float(error[first==flag,49:52].mean()) if (first==flag).any() else None) for flag in (False,True)})
        print(json.dumps(dict(stage='rollout',steps=k+1,metrics=result[str(k+1)])),flush=True)
    predicted=torch.stack(history,1);object_ids=torch.stack(objects,1)
    idx=t[:,None]+torch.arange(64,device='cuda')[None];truth=data['next_state'][idx,l[:,None],49:]
    error=(predicted[:,-1,:3]-truth[:,-1,:3]).norm(dim=-1)
    records=[]
    for j in error.argsort(descending=True)[:20].tolist():
        records.append(dict(lane=int(l[j]),start_tick=int(t[j]),final_position_error_m=float(error[j]),
            predicted_final_pv=predicted[j,-1].cpu().tolist(),true_final_pv=truth[j,-1].cpu().tolist(),
            predicted_min_ball_z=float(predicted[j,:,2].min()),
            selected_object_counts=torch.bincount(object_ids[j],minlength=4).cpu().tolist(),
            true_events={name:torch.where(masks[name][idx[j],l[j]])[0].cpu().tolist() for name in ('new_racket','new_ground','new_net')}))
    write_json(a.output/'rollout_validation.json',result)
    write_json(a.output/'worst_window_audit.json',dict(worst20=records,
        predicted_center_below_z0_windows=int((predicted[:,:,2]<0).any(1).sum()),
        native_center_below_z0_windows=int((truth[:,:,2]<0).any(1).sum())))
    torch.cuda.synchronize()
    write_json(a.output/'completed.json',dict(scope='Offline physical-prior diagnosis, not native performance',
        elapsed_s=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        optimizer_updates=0,actor_updates=0,critic_updates=0,native_steps=0,windows=len(t),adopted_for_actor=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'startup.json',dict(arguments={k:str(v) for k,v in vars(a).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:run(a)
    except BaseException as e:
        (a.output/'failure_traceback.txt').write_text(traceback.format_exc())
        write_json(a.output/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
