#!/usr/bin/env python3
"""Fit separate contact/dynamics branches, with fixed held-out lane windows."""
import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F

from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2,event_window_pools,sample_event_windows
from scripts.fit_supermimic_physical_world import model_args,validate,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=200);p.add_argument('--batch',type=int,default=128)
    args=p.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'startup.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty_diff=subprocess.check_output(['git','diff'],text=True),actor_updates=0))
    try:
        if not torch.cuda.is_available():raise RuntimeError('GPU required; no CPU-only fallback')
        torch.set_num_threads(2);torch.manual_seed(971);start=time.perf_counter()
        receipt=json.loads(args.source_receipt.read_text());source=Path(receipt['checkpoint']).parent
        for key in ('checkpoint','dataset'):
            if hashlib.sha256(Path(receipt[key]).read_bytes()).hexdigest()!=receipt[key+'_sha256']:
                raise ValueError('Changed source artifact: '+key)
        packed=torch.load(receipt['dataset'],map_location='cpu',weights_only=True);cpu=packed['tensors']
        checkpoint=torch.load(receipt['checkpoint'],map_location='cpu',weights_only=True)
        if checkpoint['schema']!=PhysicalWorldV1.schema:raise ValueError('Explicit v1 migration required')
        split=json.loads((source/'split.json').read_text());train=torch.tensor(split['training_lanes'])
        pools=event_window_pools(cpu,train,16)
        manifest=packed['source_manifest'];suite=manifest['task_contract']['suite'];path=Path(suite['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=suite['sha256']:raise ValueError('Changed launch suite')
        cases=json.loads(path.read_text())['cases']
        launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],dtype=torch.float32,device='cuda:0')
        data={k:v.to('cuda:0') for k,v in cpu.items()}
        split.update(sampling='50% uniform; 50% equal nonempty task-event/foot-transition categories',
            category_windows={k:len(v) for k,v in pools.items()},historical_event_archive_present=False)
        write_json(args.output/'split.json',split)
        write_json(args.output/'source.json',dict(receipt=receipt,source_world_updates=checkpoint['world_updates'],
            migration='Strict body/flight transfer and unchanged train-only normalization; new contact network and fresh optimizers',
            dataset_path=receipt['dataset'],source_manifest=manifest))
        source_updates=checkpoint['world_updates']
        old=PhysicalWorldV1();old.load_state_dict(checkpoint['model'],strict=True)
        model=PhysicalWorldV2(manifest['task_contract']['asset']['sha256']);model.import_v1(old);model=model.to('cuda:0')
        del packed,cpu,old,checkpoint
        dynamics_opt=torch.optim.Adam(model.body.parameters(),lr=2e-4)
        contact_opt=torch.optim.Adam(model.contact.parameters(),lr=2e-4)
        generator=torch.Generator().manual_seed(971)
        validation=torch.tensor(split['validation_selected'],device='cuda:0')
        write_json(args.output/'validation_initial.json',validate(model,data,validation,launch))
        torch.cuda.synchronize();setup_s=time.perf_counter()-start;training_start=time.perf_counter()
        sample_counts={k:0 for k in pools}
        with (args.output/'training.jsonl').open('w',buffering=1) as log:
            for update in range(1,args.updates+1):
                begin=time.perf_counter()
                # Contact calibration uses uniform native transitions from
                # training lanes only; no event-class reweighting of targets.
                times=torch.randint(len(data['state']),(args.batch*16,),generator=generator)
                lane_idx=torch.randint(len(train),(args.batch*16,),generator=generator)
                contact_pairs=torch.stack((times,train[lane_idx]),-1).to('cuda:0')
                target=data['next_foot_contact'][contact_pairs[:,0],contact_pairs[:,1]].float()
                probability=model.contact_probabilities(*model_args(data,contact_pairs,launch))
                contact_loss=F.binary_cross_entropy(probability,target)
                if not torch.isfinite(contact_loss):raise FloatingPointError('Nonfinite contact loss')
                contact_opt.zero_grad(set_to_none=True);contact_loss.backward()
                contact_grad=torch.nn.utils.clip_grad_norm_(model.contact.parameters(),10.,error_if_nonfinite=True)
                contact_opt.step();contact_opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize();contact_end=time.perf_counter()
                pairs,counts=sample_event_windows(pools,args.batch,generator,update)
                for name,count in counts.items():sample_counts[name]+=count
                pairs=pairs.to('cuda:0');t,l=pairs[:,0],pairs[:,1]
                state,_,_,feet,present,elapsed,launch_batch=model_args(data,pairs,launch)
                loss=state.new_zeros(())
                for k in range(16):
                    state,logits,present=model(state,data['final_joint_targets'][t+k,l],
                        data['observation'][t+k,l,-18:],feet,present,elapsed,launch_batch,detach_contacts=True)
                    loss=loss+model.state_loss(state,data['next_state'][t+k,l])/16
                    feet=logits.sigmoid();elapsed=elapsed+.01
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite dynamics loss')
                dynamics_opt.zero_grad(set_to_none=True);loss.backward()
                if any(p.grad is not None for p in model.contact.parameters()):
                    raise RuntimeError('Dynamics fit leaked gradients into contact parameters')
                dynamics_grad=torch.nn.utils.clip_grad_norm_(model.body.parameters(),10.,error_if_nonfinite=True)
                dynamics_opt.step();torch.cuda.synchronize()
                row=dict(update=update,dynamics_loss=float(loss),contact_loss=float(contact_loss),
                    dynamics_gradient_before_clip=float(dynamics_grad),contact_gradient_before_clip=float(contact_grad),
                    contact_fit_s=contact_end-begin,dynamics_fit_s=time.perf_counter()-contact_end,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),actor_updates=0)
                log.write(json.dumps(row)+'\n')
                if update==1 or update%10==0:print(json.dumps(row),flush=True)
        torch.cuda.synchronize();training_s=time.perf_counter()-training_start;begin=time.perf_counter()
        write_json(args.output/'validation_final.json',validate(model,data,validation,launch));torch.cuda.synchronize()
        torch.save(dict(schema=model.schema,model=model.state_dict(),dynamics_optimizer=dynamics_opt.state_dict(),
            contact_optimizer=contact_opt.state_dict(),source_world_updates=source_updates,world_updates=args.updates,
            contact_updates=args.updates,actor_updates=0,source_manifest=manifest,
            train_lanes=train,validation_lanes=split['validation_lanes']),args.output/'world.pt')
        write_json(args.output/'completed.json',dict(world_updates=args.updates,contact_updates=args.updates,
            actor_updates=0,critic_updates=0,setup_and_initial_validation_s=setup_s,training_s=training_s,
            final_validation_s=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            sampled_window_categories=sample_counts,scope='Offline physical/contact fit; no policy performance claim'))
    except BaseException as e:
        (args.output/'failure_traceback.txt').write_text(traceback.format_exc())
        write_json(args.output/'failure.json',dict(type=type(e).__name__,message=str(e)));raise


if __name__=='__main__':main()
