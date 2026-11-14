#!/usr/bin/env python3
"""One fixed 16-step ball-response fit with frozen robot/gate and geometry."""
import argparse,json,subprocess,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_window_pools,ball_error,ball_transition
from quadsmash.rl.supermimic_ball_geometry_preview_v1 import BallGeometryPreviewV1,GeometrySupportedBallV1
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2,response_mode_labels
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2,sample_event_windows
from scripts.fit_supermimic_ball_residual_v2 import verified,diagnostics
from scripts.fit_supermimic_ball_residual import validate
from scripts.fit_supermimic_physical_world import model_args,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=200)
    p.add_argument('--batch',type=int,default=16)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'startup.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True)))
    try:
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        if a.updates<1 or a.batch<2:raise ValueError('Invalid experiment size')
        torch.set_num_threads(2);torch.manual_seed(971);begin=time.perf_counter()
        receipt=json.loads(a.source_receipt.read_text());source=Path(receipt['run'])
        settings=json.loads((source/'inputs.json').read_text());inputs=settings['source_inputs']
        physicalpath=verified(receipt['physical_checkpoint'],receipt['physical_checkpoint_sha256'])
        c=torch.load(physicalpath,map_location='cpu',weights_only=True)
        base=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).cuda()
        base.load_state_dict(c['model'],strict=True)
        ballpath=verified(receipt['ball_checkpoint'],receipt['ball_checkpoint_sha256'])
        ball=BallResidualV2().cuda();ball.load_state_dict(torch.load(ballpath,map_location='cpu',weights_only=True)['model'],strict=True)
        preview=BallGeometryPreviewV1().cuda()
        if preview.geometry_hashes!=receipt['geometry_hashes'] or preview.softness_m!=receipt['surrogate_softness_m']:
            raise ValueError('Geometry preview contract changed')
        model=GeometrySupportedBallV1(ball,preview)
        for par in list(base.parameters())+list(model.parameters()):par.requires_grad_(False)
        for par in ball.response.parameters():par.requires_grad_(True)
        frozen={k:v.detach().cpu().clone() for k,v in ball.state_dict().items() if not k.startswith('response.')}
        old_response={k:v.detach().cpu().clone() for k,v in ball.response.state_dict().items()}
        optimizer=torch.optim.Adam(ball.response.parameters(),lr=2e-5)
        info=json.loads((physicalpath.parent/'source.json').read_text())
        dataset=verified(inputs['dataset'],info['receipt']['dataset_sha256'])
        raw=torch.load(dataset,map_location='cpu',weights_only=True)['tensors']
        trace=inputs['trace_receipt'];res=torch.load(verified(trace['residuals'],trace['residuals_sha256']),map_location='cpu',weights_only=True)
        lanes=torch.tensor(inputs['split']['training_lanes']);val_lanes=torch.tensor(inputs['split']['validation_lanes'])
        if set(lanes.tolist())&set(val_lanes.tolist()):raise ValueError('Split overlap')
        pools=ball_window_pools(raw,res['masks'],lanes,16)
        data={k:v.cuda() for k,v in raw.items()}
        target=res['ball_residual'].cuda()
        with torch.no_grad():
            features=torch.cat([ball_geometry_features(s,base.geometry) for s in data['state'].flatten(0,1).split(2048)]).reshape(*data['state'].shape[:2],25)
        suite=c['source_manifest']['task_contract']['suite'];suitepath=verified(suite['path'],suite['sha256'])
        cases=json.loads(suitepath.read_text())['cases']
        launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
        validation=torch.tensor(inputs['split']['validation_selected'],device='cuda')
        initial=validate(base,model,data,validation,launch)
        previous=json.loads((source/'rollout_validation.json').read_text())
        max_error=max(abs(initial[k][name]-previous[k][name]) for k in initial for name in ('ball_position_mae_m','ball_velocity_mae_m_s','joint_mae_rad'))
        if max_error>2e-6:raise ValueError('Frozen source forward mismatch: '+str(max_error))
        write_json(a.output/'validation_initial.json',initial)
        write_json(a.output/'inputs.json',dict(source_receipt=receipt,source_inputs=inputs,
            window_steps=16,window_s=.16,learning_rate=2e-5,batch=a.batch,source_forward_metric_max_error=max_error,
            optimized='ball response only; gate, normalization, geometry, robot, actor, critic frozen',
            scales={'position_m':.1,'velocity_m_s':2.},loss='0.5 * mean16 autoregressive ball MSE + 0.5 * native one-step residual MSE',
            pool_sizes={k:len(v) for k,v in pools.items()},sampling='50% uniform + 50% equal available ball event windows',
            gradient='Full 16-step state graph including frozen robot inputs; no teacher-forced states or future event inputs'))
        torch.cuda.synchronize();setup=time.perf_counter()-begin;begin=time.perf_counter()
        rng=torch.Generator().manual_seed(971);counts={k:0 for k in pools}
        with (a.output/'training.jsonl').open('w',buffering=1) as log:
            for step in range(1,a.updates+1):
                start=time.perf_counter();pairs,sampled=sample_event_windows(pools,a.batch,rng,step)
                for key,n in sampled.items():counts[key]+=n
                pairs=pairs.cuda();t,l=pairs.T
                state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
                loss=state.new_zeros(())
                for k in range(16):
                    state,feet,present=ball_transition(base,model,state,data['final_joint_targets'][t+k,l],
                        data['observation'][t+k,l,-18:],feet,present,elapsed,launch_batch)
                    elapsed=elapsed+.01
                    loss=loss+ball_error(state[:,49:],data['next_state'][t+k,l,49:])
                ar_loss=loss/16
                tt=t[:,None]+torch.arange(16,device='cuda')[None];ll=l[:,None].expand_as(tt)
                correction,_=model(features[tt,ll].flatten(0,1),data['ball_present'][tt,ll].flatten())
                native_loss=ball_error(correction,target[tt,ll].flatten(0,1))
                total=.5*(ar_loss+native_loss)
                if not torch.isfinite(total):raise FloatingPointError('Nonfinite sequence loss')
                optimizer.zero_grad(set_to_none=True);total.backward()
                gradient=torch.nn.utils.clip_grad_norm_(ball.response.parameters(),10.,error_if_nonfinite=True)
                if any(p.grad is not None for p in base.parameters()) or any(p.grad is not None for p in ball.gate.parameters()):
                    raise RuntimeError('Frozen model received parameter gradients')
                optimizer.step();torch.cuda.synchronize()
                row=dict(update=step,autoregressive_loss=float(ar_loss),native_one_step_loss=float(native_loss),
                    loss=float(total),gradient_before_clip=float(gradient),seconds=time.perf_counter()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(row)+'\n')
                if step==1 or step%10==0:print(json.dumps(row),flush=True)
        torch.cuda.synchronize();training=time.perf_counter()-begin
        if any(not torch.equal(v,ball.state_dict()[k].detach().cpu()) for k,v in frozen.items()):raise RuntimeError('Frozen state changed')
        change=sum((v-ball.response.state_dict()[k].detach().cpu()).square().sum().item() for k,v in old_response.items())**.5
        write_json(a.output/'rollout_validation.json',validate(base,model,data,validation,launch))
        flat=torch.arange(len(raw['state'])*raw['state'].shape[1]).reshape(*raw['state'].shape[:2])
        ids=flat[:,val_lanes].flatten().cuda();masks={k:v.flatten().cuda() for k,v in res['masks'].items()}
        labels=response_mode_labels(masks,target.flatten(0,1),data['ball_present'].flatten())
        write_json(a.output/'one_step_diagnostics.json',diagnostics(model,features.flatten(0,1),target.flatten(0,1),labels,data['ball_present'].flatten(),masks,ids))
        torch.save(dict(schema=ball.schema,model=ball.state_dict(),optimizer=optimizer.state_dict(),
            response_sequence_updates=a.updates,geometry_hashes=preview.geometry_hashes,
            surrogate_softness_m=preview.softness_m,source_receipt=receipt),a.output/'ball.pt')
        write_json(a.output/'completed.json',dict(response_updates=a.updates,actor_updates=0,critic_updates=0,robot_updates=0,gate_updates=0,
            source_forward_metric_max_error=max_error,frozen_state_unchanged=True,response_parameter_l2_change=change,
            setup_s=setup,training_s=training,peak_allocated_bytes=torch.cuda.max_memory_allocated(),sample_counts=counts,
            scope='Offline supervised dynamics fitting, not policy learning or native performance'))
    except BaseException as e:
        (a.output/'failure_traceback.txt').write_text(traceback.format_exc());write_json(a.output/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
