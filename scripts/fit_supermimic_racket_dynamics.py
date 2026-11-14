#!/usr/bin/env python3
"""One robot-only multistep fit with explicit racket pose/velocity supervision."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state,racket_loss,robot_loss,approach_window_pools
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,GeometrySupportedBallV1
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2,sample_event_windows
from scripts.fit_supermimic_ball_residual_v2 import verified
from scripts.fit_supermimic_ball_residual import validate
from scripts.fit_supermimic_physical_world import model_args,write_json


@torch.no_grad()
def geometry_validation(base,ball,data,pairs,launch,masks):
    t,l=pairs.T;state,_,_,feet,present,elapsed,lb=model_args(data,pairs,launch)
    metrics={};events=[]
    for k in range(64):
        hit=masks['new_racket'][t+k,l]
        current=ball_geometry_features(state,base.geometry)
        true_current=ball_geometry_features(data['state'][t+k,l],base.geometry)
        for j in torch.where(hit)[0].tolist():
            events.append(dict(window_index=j,lane=int(l[j]),start_tick=int(t[j]),offset=k,
                local_ball_error_m=float((current[j,6:9]-true_current[j,6:9]).norm()),
                relative_velocity_error_m_s=float((current[j,9:12]-true_current[j,9:12]).norm())))
        state,feet,present=ball_transition(base,ball,state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],feet,present,elapsed,lb)
        elapsed=elapsed+.01
        if k+1 not in (1,8,16,32,64):continue
        pred=racket_state(state,base.geometry);truth=racket_state(data['next_state'][t+k,l],base.geometry)
        e=(pred-truth).abs();dist=(pred[:,:3]-truth[:,:3]).norm(dim=-1)
        rotation=pred[:,3:12].reshape(-1,3,3).transpose(-1,-2)@truth[:,3:12].reshape(-1,3,3)
        angle=((rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1).acos()*180/torch.pi
        def stats(mask):
            n=int(mask.sum())
            if not n:return {'count':0}
            return dict(count=n,position_mae_m=float(e[mask,:3].mean()),position_euclidean_p95_m=float(torch.quantile(dist[mask],.95)),
                orientation_mean_deg=float(angle[mask].mean()),velocity_mae_m_s=float(e[mask,12:15].mean()),omega_mae_rad_s=float(e[mask,15:18].mean()))
        # Diagnostics only: within 16 ticks before a native impact, with same episode.
        offsets=torch.arange(16,device=t.device);idx=(t[:,None]+k+offsets[None]).clamp_max(len(data['state'])-1)
        near=(masks['new_racket'][idx,l[:,None]] & (data['episode'][idx,l[:,None]]==data['episode'][t+k,l,None])).any(1)
        metrics[str(k+1)]={'all':stats(torch.ones(len(t),device=t.device,dtype=torch.bool)),'approaching_native_impact':stats(near)}
    return dict(metrics=metrics,first_impact_records=events)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--source-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=200);p.add_argument('--batch',type=int,default=16)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'startup.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_diff=subprocess.check_output(['git','diff'],text=True)))
    try:
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        if a.updates<1 or a.batch<2:raise ValueError('Positive updates and batch >=2 required')
        torch.set_num_threads(2);torch.manual_seed(971);begin=time.perf_counter()
        inputs=json.loads((a.source_run/'inputs.json').read_text());r=inputs['source_receipt'];old=inputs['source_inputs']
        physicalpath=verified(r['physical_checkpoint'],r['physical_checkpoint_sha256'])
        c=torch.load(physicalpath,map_location='cpu',weights_only=True)
        base=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).cuda();base.load_state_dict(c['model'],strict=True)
        br=json.loads(a.source_receipt.read_text());bp=verified(a.source_run/'ball.pt',br['checkpoint_sha256'])
        bc=torch.load(bp,map_location='cpu',weights_only=True)
        ball=BallResidualV2().cuda();ball.load_state_dict(bc['model'],strict=True)
        preview=BallGeometryPreviewV1().cuda()
        if preview.geometry_hashes!=bc['geometry_hashes']:raise ValueError('Changed geometry')
        model=GeometrySupportedBallV1(ball,preview)
        base.requires_grad_(False);model.requires_grad_(False);base.body.requires_grad_(True)
        frozen={k:v.detach().cpu().clone() for k,v in base.state_dict().items() if not k.startswith('body.')}
        frozen_ball={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        old_body={k:v.detach().cpu().clone() for k,v in base.body.state_dict().items()}
        metadata=json.loads((physicalpath.parent/'source.json').read_text())
        dataset=verified(old['dataset'],metadata['receipt']['dataset_sha256'])
        raw=torch.load(dataset,map_location='cpu',weights_only=True)['tensors']
        trace=old['trace_receipt'];res=torch.load(verified(trace['residuals'],trace['residuals_sha256']),map_location='cpu',weights_only=True)
        lanes=torch.tensor(old['split']['training_lanes']);pools=approach_window_pools(raw,res['masks'],lanes,32)
        if set(lanes.tolist())&set(old['split']['validation_lanes']):raise ValueError('Split overlap')
        data={k:v.cuda() for k,v in raw.items()};masks={k:v.cuda() for k,v in res['masks'].items()};del raw,res
        with torch.no_grad():
            target_racket=torch.cat([racket_state(s,base.geometry) for s in data['next_state'].flatten(0,1).split(2048)]).reshape(*data['state'].shape[:2],18)
        suite=c['source_manifest']['task_contract']['suite'];cases=json.loads(verified(suite['path'],suite['sha256']).read_text())['cases']
        launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
        val=torch.tensor(old['split']['validation_selected'],device='cuda')
        initial=validate(base,model,data,val,launch);previous=json.loads((a.source_run/'rollout_validation.json').read_text())
        diff=max(abs(initial[s][k]-previous[s][k]) for s in initial for k in ('ball_position_mae_m','ball_velocity_mae_m_s','joint_mae_rad'))
        if diff>2e-6:raise ValueError('Initial source prediction mismatch '+str(diff))
        write_json(a.output/'validation_initial.json',initial)
        write_json(a.output/'racket_initial.json',geometry_validation(base,model,data,val,launch,masks))
        write_json(a.output/'inputs.json',dict(source_run=str(a.source_run),physical_checkpoint=str(physicalpath),physical_checkpoint_sha256=r['physical_checkpoint_sha256'],
            ball_checkpoint=str(bp),ball_checkpoint_sha256=br['checkpoint_sha256'],dataset=str(dataset),dataset_sha256=metadata['receipt']['dataset_sha256'],
            source_inputs=old,split=old['split'],geometry_hashes=preview.geometry_hashes,learning_rate=2e-5,batch=a.batch,window_steps=32,window_s=.32,
            optimized='robot dynamics body only; all ball/contact/normalization/geometry/actor/critic parameters frozen',
            loss='mean32(robot_loss + racket_loss); fixed physical scales in source',
            sampling='50% uniform, 50% first racket event in second half of window; native labels only for supervised sampling',
            pool_sizes={k:len(v) for k,v in pools.items()},source_metric_max_error=diff))
        optimizer=torch.optim.Adam(base.body.parameters(),lr=2e-5)
        torch.cuda.synchronize();setup=time.perf_counter()-begin;begin=time.perf_counter();rng=torch.Generator().manual_seed(971)
        counts={k:0 for k in pools}
        with (a.output/'training.jsonl').open('w',buffering=1) as log:
            for step in range(1,a.updates+1):
                start=time.perf_counter();pairs,sampled=sample_event_windows(pools,a.batch,rng,step)
                for key,n in sampled.items():counts[key]+=n
                pairs=pairs.cuda();t,l=pairs.T;state,_,_,feet,present,elapsed,lb=model_args(data,pairs,launch)
                robot=state.new_zeros(());racket=state.new_zeros(())
                for k in range(32):
                    state,feet,present=ball_transition(base,model,state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],feet,present,elapsed,lb)
                    elapsed=elapsed+.01
                    robot=robot+robot_loss(state,data['next_state'][t+k,l])/32
                    racket=racket+racket_loss(racket_state(state,base.geometry),target_racket[t+k,l])/32
                loss=robot+racket
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite dynamics loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if any(p.grad is not None for p in model.parameters()) or any(p.grad is not None for p in base.contact.parameters()):raise RuntimeError('Frozen parameter gradient')
                grad=torch.nn.utils.clip_grad_norm_(base.body.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                row=dict(update=step,loss=float(loss),robot_loss=float(robot),racket_loss=float(racket),gradient_before_clip=float(grad),
                    seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row)+'\n')
                if step==1 or step%10==0:print(json.dumps(row),flush=True)
        torch.cuda.synchronize();training=time.perf_counter()-begin
        if any(not torch.equal(v,base.state_dict()[k].cpu()) for k,v in frozen.items()):raise ValueError('Frozen robot/contact state changed')
        if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen_ball.items()):raise ValueError('Frozen ball state changed')
        change=sum((v-base.body.state_dict()[k].detach().cpu()).square().sum().item() for k,v in old_body.items())**.5
        write_json(a.output/'validation_final.json',validate(base,model,data,val,launch))
        write_json(a.output/'racket_final.json',geometry_validation(base,model,data,val,launch,masks))
        torch.save(dict(schema=base.schema,model=base.state_dict(),dynamics_optimizer=optimizer.state_dict(),
            source_manifest=c['source_manifest'],source_world_updates=c.get('source_world_updates',0),
            world_updates=c['world_updates']+a.updates,racket_dynamics_updates=a.updates,
            source_checkpoint_sha256=r['physical_checkpoint_sha256'],source_ball_sha256=br['checkpoint_sha256']),a.output/'world.pt')
        write_json(a.output/'completed.json',dict(robot_updates=a.updates,actor_updates=0,critic_updates=0,ball_updates=0,contact_updates=0,native_steps=0,
            frozen_state_unchanged=True,robot_parameter_l2_change=change,setup_s=setup,training_s=training,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),sample_counts=counts,source_metric_max_error=diff,
            scope='Offline supervised dynamics fit; no native success claim'))
    except BaseException as e:
        (a.output/'failure_traceback.txt').write_text(traceback.format_exc());write_json(a.output/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
