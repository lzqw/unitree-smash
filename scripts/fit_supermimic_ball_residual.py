#!/usr/bin/env python3
"""Fit a separate geometry-conditioned ball response on fixed native data."""
import argparse,hashlib,json,subprocess,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v1 import BallResidualV1,ball_geometry_features
from scripts.fit_supermimic_physical_world import model_args,write_json
from scripts.audit_supermimic_physical_world_fit import classification


@torch.no_grad()
def validate(base,ball,data,pairs,launch):
    t,l=pairs.T;state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
    result={}
    for k in range(64):
        prior,_=base.prior(state,present,elapsed,launch_batch)
        correction,_=ball(ball_geometry_features(state,base.geometry),present)
        nxt,logits,active=base(state,data['final_joint_targets'][t+k,l],data['observation'][t+k,l,-18:],
                              feet,present,elapsed,launch_batch)
        state=torch.cat((nxt[:,:49],prior[:,49:]+correction),-1)
        feet=logits.sigmoid();present=active;elapsed=elapsed+.01
        if k+1 not in (1,8,16,64):continue
        err=(state-data['next_state'][t+k,l]).abs();dist=err[:,49:52].norm(dim=-1)
        idx=t[:,None]+torch.arange(k+1,device=t.device)[None]
        events=data['next_event_state'][idx,l[:,None]]>data['event_state'][idx,l[:,None]]
        hit=events[:,:,[1,6]].any(-1).any(-1)
        if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite ball rollout')
        result[str(k+1)]=dict(joint_mae_rad=float(err[:,13:31].mean()),ball_position_mae_m=float(err[:,49:52].mean()),
            ball_velocity_mae_m_s=float(err[:,52:55].mean()),ball_position_p95_euclidean_m=float(torch.quantile(dist,.95)),
            foot_accuracy=float(((feet>=.5)==data['next_foot_contact'][t+k,l]).float().mean()),
            first_contact_group={str(flag):dict(count=int((hit==flag).sum()),
                ball_position_mae_m=float(err[hit==flag,49:52].mean()) if (hit==flag).any() else None) for flag in (False,True)})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-receipt',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=600)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'startup.json',dict(argv={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True)))
    try:
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        torch.set_num_threads(2);torch.manual_seed(971);begin=time.perf_counter()
        receipt=json.loads(args.source_receipt.read_text());source=Path(receipt['checkpoint']).parent
        if hashlib.sha256(Path(receipt['checkpoint']).read_bytes()).hexdigest()!=receipt['checkpoint_sha256']:raise ValueError('Changed source model')
        source_info=json.loads((source/'source.json').read_text());data_path=Path(source_info['dataset_path'])
        if hashlib.sha256(data_path.read_bytes()).hexdigest()!=source_info['receipt']['dataset_sha256']:raise ValueError('Changed dataset')
        d=torch.load(data_path,map_location='cpu',weights_only=True)['tensors']
        residuals=torch.load(args.trace/'ball_residuals.pt',map_location='cpu',weights_only=True)
        c=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=True);split=json.loads((source/'split.json').read_text())
        if c['schema']!=PhysicalWorldV2.schema:raise ValueError('Expected frozen physical v2')
        base=PhysicalWorldV2(c['source_manifest']['task_contract']['asset']['sha256']).cuda();base.load_state_dict(c['model'],strict=True)
        for parameter in base.parameters():parameter.requires_grad_(False)
        train=torch.tensor(split['training_lanes']);val=torch.tensor(split['validation_lanes'])
        flat=torch.arange(d['state'].shape[0]*128).reshape(-1,128);train_ids=flat[:,train].flatten();val_ids=flat[:,val].flatten()
        data={k:v.cuda() for k,v in d.items()};feature=[]
        with torch.no_grad():
            for state in data['state'].flatten(0,1).split(2048):feature.append(ball_geometry_features(state,base.geometry))
        feature=torch.cat(feature);target=residuals['ball_residual'].flatten(0,1).cuda()
        present=d['ball_present'].flatten().cuda()
        # These are residual labels, not native collision certification.
        nonfree=((target[:,3:].norm(dim=-1)>.001)|(target[:,:3].norm(dim=-1)>.00001))&present
        ball=BallResidualV1().cuda();ball.normalize_from_training(feature[train_ids],target[train_ids])
        gate_opt=torch.optim.Adam(ball.gate.parameters(),lr=2e-4);response_opt=torch.optim.Adam(ball.response.parameters(),lr=2e-4)
        pools={'nonfree':train_ids[nonfree[train_ids].cpu()]}
        for name in ('new_racket','new_ground','new_net'):
            ids=train_ids[residuals['masks'][name].flatten()[train_ids]]
            if len(ids):pools[name]=ids
        suite=c['source_manifest']['task_contract']['suite'];path=Path(suite['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=suite['sha256']:raise ValueError('Changed suite')
        cases=json.loads(path.read_text())['cases'];launch=torch.tensor([x['launch']['position_m']+x['launch']['velocity_mps'] for x in cases],device='cuda')
        validation=torch.tensor(split['validation_selected'],device='cuda')
        write_json(args.output/'inputs.json',dict(source_receipt=receipt,dataset=str(data_path),trace=str(args.trace),
            trace_residual_sha256=hashlib.sha256((args.trace/'ball_residuals.pt').read_bytes()).hexdigest(),
            split=split,response_pool_sizes={k:len(v) for k,v in pools.items()},residual_thresholds={'velocity_norm_m_s':.001,'position_norm_m':.00001}))
        rng=torch.Generator().manual_seed(971)
        def sample(ids,n):return ids[torch.randint(len(ids),(n,),generator=rng)].cuda()
        torch.cuda.synchronize();setup_s=time.perf_counter()-begin;begin=time.perf_counter()
        with (args.output/'training.jsonl').open('w',buffering=1) as log:
            for update in range(1,args.updates+1):
                ids=sample(train_ids,2048);x=ball.normalized(feature[ids]);label=nonfree[ids].float()
                gate_loss=F.binary_cross_entropy_with_logits(ball.gate(x).squeeze(-1),label)
                gate_opt.zero_grad(set_to_none=True);gate_loss.backward()
                gg=torch.nn.utils.clip_grad_norm_(ball.gate.parameters(),10.,error_if_nonfinite=True);gate_opt.step()
                # Equal response categories prevent resting-ground samples
                # from overwhelming the short racket/net impulse transitions.
                ids=torch.cat([sample(pool,128) for pool in pools.values()])
                prediction=ball.response(ball.normalized(feature[ids]))
                response_loss=((prediction-target[ids]/ball.residual_scale).square()).mean()
                if not torch.isfinite(response_loss+gate_loss):raise FloatingPointError('Nonfinite ball fit')
                response_opt.zero_grad(set_to_none=True);response_loss.backward()
                rg=torch.nn.utils.clip_grad_norm_(ball.response.parameters(),10.,error_if_nonfinite=True);response_opt.step()
                row=dict(update=update,gate_loss=float(gate_loss),response_loss=float(response_loss),
                    gate_gradient_before_clip=float(gg),response_gradient_before_clip=float(rg),actor_updates=0)
                log.write(json.dumps(row)+'\n')
                if update==1 or update%50==0:print(json.dumps(row),flush=True)
        torch.cuda.synchronize();fit_s=time.perf_counter()-begin
        with torch.no_grad():
            delta,prob=ball(feature[val_ids],present[val_ids]);truth=target[val_ids]
            metrics={'nonfree_gate':classification(prob,nonfree[val_ids]),'by_group':{}}
            for name,mask in residuals['masks'].items():
                selected=mask.flatten()[val_ids]
                e=(delta-truth)[selected]
                metrics['by_group'][name]=dict(count=len(e),position_mae_m=float(e[:,:3].abs().mean()) if len(e) else None,
                    velocity_mae_m_s=float(e[:,3:].abs().mean()) if len(e) else None)
        write_json(args.output/'one_step_validation.json',metrics)
        write_json(args.output/'rollout_validation.json',validate(base,ball,data,validation,launch))
        torch.save(dict(schema=ball.schema,model=ball.state_dict(),gate_optimizer=gate_opt.state_dict(),
            response_optimizer=response_opt.state_dict(),updates=args.updates,source_receipt=receipt),args.output/'ball.pt')
        write_json(args.output/'completed.json',dict(ball_gate_updates=args.updates,ball_response_updates=args.updates,
            actor_updates=0,critic_updates=0,robot_updates=0,setup_s=setup_s,fit_s=fit_s,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),scope='Recorded-control model validation; no native performance result'))
    except BaseException as e:
        (args.output/'failure_traceback.txt').write_text(traceback.format_exc());write_json(args.output/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
