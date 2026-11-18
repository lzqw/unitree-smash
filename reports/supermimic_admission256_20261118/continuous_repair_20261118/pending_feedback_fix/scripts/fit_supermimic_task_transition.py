#!/usr/bin/env python3
"""One bounded GPU fit of missing task memory, with held-out event diagnostics.

No actor optimizer, no native step, no success score. Future native fields
are supervision only. A 16-transition window carries predicted memory,
while robot transitions are teacher-forced during this task-only fit.
"""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import torch
from quadsmash.rl.supermimic_task_memory_schema_v1 import TaskMemorySchemaV1,MemoryField
from quadsmash.rl.supermimic_task_transition_model_v1 import NativeTaskTransitionSurrogateV1
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows,split_lanes
from quadsmash.rl.supermimic_physical_world_v2 import sample_event_windows
from quadsmash.rl.supermimic_event_hazard_v1 import advance_event_latches
from quadsmash.rl.supermimic_data_split_v1 import explicit_lane_split


def metrics(probability,label,eligible):
    p=probability[eligible];y=label[eligible].bool();selected=p>=.5
    tp=int((selected&y).sum());fp=int((selected&~y).sum());fn=int((~selected&y).sum())
    return dict(count=len(y),positives=int(y.sum()),tp=tp,fp=fp,fn=fn,
                mean_positive_probability=float(p[y].mean()) if y.any() else None,
                mean_negative_probability=float(p[~y].mean()) if (~y).any() else None,
                brier=float((p-y.float()).square().mean()),
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split-manifest',type=Path,help='Explicit complete lane partition; use the same heldout cases as physical/ball models')
    parser.add_argument('--task-checkpoint',type=Path,help='Compatible same-data/split model initialization; fresh optimizer, not full resume')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261114)
    updates,batch,steps,lr=512,256,16,2e-4
    receipt=json.loads((args.data.parent/'completed.json').read_text())
    sha=hashlib.sha256(args.data.read_bytes()).hexdigest()
    if sha!=receipt['sha256']:raise ValueError('Packed data hash mismatch')
    package=torch.load(args.data,map_location='cpu',weights_only=True)
    raw=package['tensors']
    if raw['state'].shape[:2] != (800,128) or not bool(package['first_done'].all()):
        raise ValueError('Task fit requires complete original128 first episodes over 800 ticks')
    train,val=(explicit_lane_split(json.loads(args.split_manifest.read_text()),128) if args.split_manifest
               else split_lanes(128,16,20261114))
    schema=TaskMemorySchemaV1([MemoryField(**dict(f,shape=tuple(f['shape']))) for f in package['task_schema']['fields']])
    model=NativeTaskTransitionSurrogateV1(schema,raw['torque'].shape[-1]).cuda()
    initialized=None
    if args.task_checkpoint:
        initialized=torch.load(args.task_checkpoint,map_location='cpu',weights_only=True)
        source_config=initialized['config']
        if (initialized['schema']!=model.schema or initialized['task_schema']!=schema.manifest()
                or source_config['data_sha256']!=sha or source_config['train_lanes']!=train.tolist()
                or source_config['validation_lanes']!=val.tolist()):
            raise ValueError('Task initialization changes data, schema or independent holdout split')
        model.load_state_dict(initialized['model'],strict=True)
    needed=('state','next_state','targets','previous_action','feet','elapsed','values','valid',
            'next_values','next_valid','event_state','next_event_state','torque','terminated','complete_reward')
    data={k:raw[k].cuda() for k in needed}
    if initialized is None:
        model.set_normalization(data['values'][:,train].flatten(0,1),data['valid'][:,train].flatten(0,1))
    windows=valid_windows(raw['episode'],raw['terminated'],raw['timeout'],steps)
    allowed=torch.zeros(128,dtype=torch.bool);allowed[train]=True;windows&=allowed[None]
    pools={'uniform':windows.nonzero()}
    names=('jump','hit','cross','target','recovery','body','frame','fault')
    for i,name in enumerate(names):
        event=(raw['next_event_state'][...,i]>raw['event_state'][...,i]).unfold(0,steps,1).any(-1)&windows
        if event.any():pools[name]=event.nonzero()
    # Time limits and complete5 are distinct native labels. A valid window
    # may end at a terminal, never cross it. Explicit pools prevent the sole
    # terminal per episode from disappearing in uniform800-tick sampling.
    for name,event in dict(episode_end=raw['terminated']|raw['timeout'],
                           complete_terminal=raw['complete_reward'].bool()).items():
        mask=event.unfold(0,steps,1).any(-1)&windows
        if mask.any():pools[name]=mask.nonzero()
    # Native-frequency prior, not a made-up reward or label. This keeps a
    # randomly initialized rare-event head from starting at 0.5 per 10 ms.
    paid=data['event_state'][:,train];after=data['next_event_state'][:,train]
    frequencies=((after-paid).sum((0,1))/(1-paid).sum((0,1)).clamp_min(1)).clamp(1/len(train)/800,1-1e-5)
    if initialized is None:
        with torch.no_grad():
            model.net.net[-1].bias[-10:-2].copy_(torch.logit(frequencies))
            model.net.net[-1].bias[-2].fill_(float(torch.logit(data['terminated'][:,train].float().mean().clamp_min(1/len(train)/800))))
            model.net.net[-1].bias[-1].fill_(float(torch.logit(data['complete_reward'][:,train].float().mean().clamp_min(1/len(train)/800))))
    config=dict(data_sha256=sha,updates=updates,batch=batch,world_steps=steps,lr=lr,
                actor_updates=0,physical_model_updates=0,seed=20261114,train_lanes=train.tolist(),validation_lanes=val.tolist(),
                sampling='50% uniform windows, 50% cyclic native event windows',
                split_manifest_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest() if args.split_manifest else None,
                source_manifest=package.get('source_manifest'),initialization='random; no legacy task parameters or optimizer restored',
                supervision_contract=model.supervision_contract,
                feedback_contract=schema.feedback_contract,
                physical_input='teacher-forced native transition; task memory autoregressive',
                event_prior=frequencies.tolist(),window_pools={k:len(v) for k,v in pools.items()})
    if initialized is not None:
        config.update(initialization='compatible same-data/split task model; fresh Adam explicitly, no full resume',
                      initial_checkpoint=str(args.task_checkpoint.resolve()),
                      initial_checkpoint_sha256=hashlib.sha256(args.task_checkpoint.read_bytes()).hexdigest(),
                      initial_task_updates=int(model.fit_updates))
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    optimizer=torch.optim.Adam(model.parameters(),lr=lr);generator=torch.Generator().manual_seed(20261114)
    started=time.perf_counter()
    for update in range(updates):
        selected,_=sample_event_windows(pools,batch,generator,update)
        t,l=selected.to('cuda').unbind(1)
        values,validity=data['values'][t,l],data['valid'][t,l]
        loss=values.new_zeros(())
        for j in range(steps):
            at=t+j
            out=model.forward_values(data['state'][at,l],data['next_state'][at,l],data['targets'][at,l],
                data['previous_action'][at,l],data['feet'][at,l],data['elapsed'][at,l],values,validity)
            loss=loss+model.supervision_loss_values(out,data['next_values'][at,l],data['next_valid'][at,l],
                data['event_state'][at,l],data['next_event_state'][at,l],data['torque'][at,l],
                data['terminated'][at,l],data['complete_reward'][at,l])
            values,validity=model.memory_schema.feedback_values(out['numeric'],out['valid_logits'].sigmoid())
        loss=loss/steps
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite task-memory sequence loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();model.fit_updates.add_(1)
        if update%16==0 or update+1==updates:
            row=dict(update=update+1,loss=float(loss),gradient=float(grad),seconds=time.perf_counter()-started,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated())
            print(json.dumps(row),flush=True)
            with (args.output/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    destination=args.output/'task.pt'
    torch.save(dict(schema=model.schema,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    task_schema=schema.manifest(),config=config,task_updates=int(model.fit_updates)),destination)
    probabilities=[];labels=[];eligible=[];memory_errors=[];terminal=[]
    model.eval()
    with torch.no_grad():
        indices=torch.cartesian_prod(torch.arange(800),val).cuda()
        for chunk in indices.split(2048):
            t,l=chunk.unbind(1)
            out=model.forward_values(data['state'][t,l],data['next_state'][t,l],data['targets'][t,l],
                data['previous_action'][t,l],data['feet'][t,l],data['elapsed'][t,l],data['values'][t,l],data['valid'][t,l])
            probabilities.append(out['hazards'].sigmoid().cpu())
            labels.append(data['next_event_state'][t,l].cpu());eligible.append((data['event_state'][t,l]==0).cpu())
            memory_errors.append((((out['numeric']-data['next_values'][t,l])/model.memory_scale).square()*data['next_valid'][t,l]).cpu())
            terminal.append(out['terminal_logit'].sigmoid().cpu())
    probabilities,labels,eligible=map(torch.cat,(probabilities,labels,eligible))
    result=dict(config=config,checkpoint=str(destination),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                actual_optimizer_updates=updates,
                task_updates=int(model.fit_updates),actor_updates=0,
                validation_events={name:metrics(probabilities[:,i],labels[:,i],eligible[:,i]) for i,name in enumerate(names)},
                validation_normalized_memory_mse=float(torch.cat(memory_errors).mean()),
                validation_termination_probability_mean=float(torch.cat(terminal).mean()),
                elapsed_seconds=time.perf_counter()-started,
                limitation='Task surrogate diagnostics, NOT native task performance or actor-update authorization')
    (args.output/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
