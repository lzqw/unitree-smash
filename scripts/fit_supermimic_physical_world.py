#!/usr/bin/env python3
"""GPU physical-model experiment on immutable native data; never updates actor.

Recorded final controls define open-loop dynamics supervision, not actor
imagination. Complete lanes are held out before normalization or optimization.
"""
import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

from quadsmash.rl.supermimic_physical_dataset_v1 import pack_archive, valid_windows, split_lanes
from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def choose(pool, count, generator):
    if not len(pool):
        raise ValueError('Empty supervised window pool')
    return pool[torch.randint(len(pool),(count,),generator=generator)]


def window_pools(data, lanes, steps):
    valid=valid_windows(data['episode'],data['terminated'],data['timeout'],steps)
    allowed=torch.zeros(valid.shape[1],dtype=torch.bool);allowed[lanes]=True
    valid &= allowed[None]
    event=(data['next_event_state']>data['event_state']).any(-1)
    event |= (data['next_foot_contact']!=data['foot_contact']).any(-1)
    rare=event.unfold(0,steps,1).any(-1)
    return valid.nonzero(), (valid&rare).nonzero()


def model_args(data, pairs, launch):
    t,l=pairs[:,0],pairs[:,1]
    return (data['state'][t,l], data['final_joint_targets'][t,l],
            data['observation'][t,l,-18:],data['foot_contact'][t,l],
            data['ball_present'][t,l],data['elapsed'][t,l],launch[l])


@torch.no_grad()
def validate(model,data,pairs,launch):
    t,l=pairs[:,0],pairs[:,1]
    state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
    phase=data['observation'][t,l,770:775].argmax(-1)
    result={}
    for k in range(64):
        state,logits,present=model(state,data['final_joint_targets'][t+k,l],
            data['observation'][t+k,l,-18:],feet,present,elapsed,launch_batch)
        feet=logits.sigmoid();elapsed=elapsed+model.dt
        if k+1 not in (1,8,16,64):continue
        target=data['next_state'][t+k,l]
        error=(state-target).abs()
        norms=torch.linalg.vector_norm(state[:,49:52]-target[:,49:52],dim=-1)
        def stats(mask):
            if not mask.any():return {'count':0}
            return dict(count=int(mask.sum()),joint_mae_rad=float(error[mask,13:31].mean()),
                root_position_mae_m=float(error[mask,:3].mean()),
                root_velocity_mae_m_s=float(error[mask,7:10].mean()),
                ball_position_mae_m=float(error[mask,49:52].mean()),
                ball_position_p95_euclidean_m=float(torch.quantile(norms[mask],.95)),
                ball_velocity_mae_m_s=float(error[mask,52:55].mean()),
                foot_probability_mae=float((feet[mask]-data['next_foot_contact'][t+k,l][mask].float()).abs().mean()))
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite held-out autoregressive physical state')
        result[str(k+1)]={'all':stats(torch.ones(len(t),dtype=torch.bool,device=t.device)),
                         'by_initial_phase':{str(p):stats(phase==p) for p in range(5)}}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=200);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--seed',type=int,default=971);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'startup.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True),actor_updates=0))
    try:
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('This experiment requires a GPU; no CPU-only training fallback')
        torch.set_num_threads(2);torch.manual_seed(args.seed)
        start=time.perf_counter();packed=pack_archive(args.archive);cpu=packed['tensors']
        torch.save(packed,args.output/'physical_dataset.pt')
        manifest=packed['source_manifest'];suite=manifest['task_contract']['suite']
        path=Path(suite['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=suite['sha256']:
            raise ValueError('Launch suite hash does not match recorded native environment')
        cases=json.loads(path.read_text())['cases']
        launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],dtype=torch.float32)
        if manifest['num_envs']!=128 or len(cases)!=128:
            raise ValueError('Expected original128 native archive and lane order')
        train,val=split_lanes(128,32,args.seed)
        pool,rare=window_pools(cpu,train,16);vp,_=window_pools(cpu,val,64)
        generator=torch.Generator().manual_seed(args.seed)
        # Equal per-start-phase coverage is diagnostic, not a task success rate.
        phase=cpu['observation'][vp[:,0],vp[:,1],770:775].argmax(-1)
        selected=[choose(vp[phase==k],64,generator) for k in range(5) if (phase==k).any()]
        validation=torch.cat(selected)
        write_json(args.output/'split.json',dict(training_lanes=train.tolist(),validation_lanes=val.tolist(),
            training_windows=len(pool),current_event_windows=len(rare),validation_windows=len(vp),
            validation_selected=validation.tolist(),window_steps=16,
            sampling='50% uniform + 50% current event windows; no historical archive yet',
            scope='Held-out lanes in fixed development panel, not unseen-initial-state generalization',
            known_launch_input='Environment launch position/velocity only; never actor observation or future outcome'))
        device=torch.device(args.device)
        data={k:v.to(device) for k,v in cpu.items()};launch=launch.to(device)
        del packed,cpu
        model=PhysicalWorldV1().to(device)
        # Train-only normalization includes each native transition once.
        features=[];priors=[];targets=[]
        train_pairs=torch.stack(torch.meshgrid(torch.arange(len(data['state'])),train,indexing='ij'),-1).reshape(-1,2)
        for batch in train_pairs.split(2048):
            b=batch.to(device);inputs=model_args(data,b,launch)
            with torch.no_grad():
                features.append(model.features(*inputs));priors.append(model.prior(inputs[0],inputs[4],inputs[5],inputs[6])[0])
                targets.append(data['next_state'][b[:,0],b[:,1]])
        model.set_normalization(torch.cat(features),torch.cat(priors),torch.cat(targets))
        del features,priors,targets
        torch.cuda.synchronize();setup_s=time.perf_counter()-start
        optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
        validation=validation.to(device)
        write_json(args.output/'validation_initial.json',validate(model,data,validation,launch))
        torch.cuda.synchronize();training_start=time.perf_counter()
        with (args.output/'training.jsonl').open('w',buffering=1) as log:
            for update in range(1,args.updates+1):
                begin=time.perf_counter()
                count=args.batch//2 if len(rare) else 0
                pairs=torch.cat((choose(pool,args.batch-count,generator),choose(rare,count,generator))) if count else choose(pool,args.batch,generator)
                pairs=pairs.to(device);t,l=pairs[:,0],pairs[:,1]
                state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
                loss=state.new_zeros(())
                for k in range(16):
                    state,logits,present=model(state,data['final_joint_targets'][t+k,l],
                        data['observation'][t+k,l,-18:],feet,present,elapsed,launch_batch)
                    loss=loss+model.state_loss(state,data['next_state'][t+k,l])
                    loss=loss+.1*F.binary_cross_entropy_with_logits(logits,data['next_foot_contact'][t+k,l].float())
                    feet=logits.sigmoid();elapsed=elapsed+.01
                loss=loss/16
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical world loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                record=dict(update=update,loss=float(loss),grad_before_clip=float(grad),elapsed_s=time.perf_counter()-begin,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),actor_updates=0)
                log.write(json.dumps(record)+'\n')
                if update==1 or update%10==0:print(json.dumps(record),flush=True)
        torch.cuda.synchronize();training_s=time.perf_counter()-training_start
        begin=time.perf_counter();metrics=validate(model,data,validation,launch);torch.cuda.synchronize()
        write_json(args.output/'validation_final.json',metrics)
        torch.save(dict(schema=model.schema,model=model.state_dict(),optimizer=optimizer.state_dict(),world_updates=args.updates,
                        actor_updates=0,source_manifest=manifest,train_lanes=train,validation_lanes=val),args.output/'world.pt')
        write_json(args.output/'completed.json',dict(world_updates=args.updates,actor_updates=0,critic_updates=0,
            setup_and_transfer_s=setup_s,training_s=training_s,final_validation_s=time.perf_counter()-begin,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),scope='Physical dynamics fit only; no policy performance claim'))
    except BaseException as e:
        (args.output/'failure_traceback.txt').write_text(traceback.format_exc())
        write_json(args.output/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise


if __name__=='__main__':main()
