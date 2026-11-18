#!/usr/bin/env python3
"""Offline fit of an opt-in swept-racket ball candidate; never updates Actor/PhysX.

No native paired perturbations are performed. The resulting candidate must be
validated physically before it is allowed to support policy training.
"""
import argparse
import hashlib
import json
import time
import subprocess
import sys
from pathlib import Path
import torch
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1,preimpact_hard_negatives,contact_neighborhood_pools
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_wm_validation_v1 import event_aligned_windows


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def one_step(base,ball,sweep,data,pairs,launch):
    t,l=pairs.to(launch.device).unbind(1)
    state=data['state'][t,l];target=data['targets'][t,l]
    other=[data[k][t,l] for k in ('previous_action','feet','present','elapsed')]
    prediction,_,_=ball_transition(base,ball,state,target,*other,launch[l],action_model=sweep)
    truth=data['next_state'][t,l,49:55]
    return prediction[:,49:55],truth


def metrics(base,ball,sweep,data,metadata,lanes,launch,*,maximum=128):
    groups=event_aligned_windows(metadata,lanes,horizon=64,offsets=(-1,0,1),maximum=maximum)
    out={}
    with torch.no_grad():
        for offset,entry in groups.items():
            pairs=entry['starts']
            if not len(pairs):
                out[offset]=dict(count=0,velocity_rmse=None,forward_sign_error=None)
                continue
            pred,true=one_step(base,ball,sweep,data,pairs,launch)
            out[offset]=dict(count=len(pairs),
                velocity_rmse=float((pred[:,3:]-true[:,3:]).square().mean().sqrt()),
                forward_sign_error=float(((pred[:,3]>0)!=(true[:,3]>0)).float().mean()),
                position_rmse=float((pred[:,:3]-true[:,:3]).square().mean().sqrt()))
    return out


@torch.no_grad()
def autoregressive_metrics(base,ball,sweep,data,metadata,lanes,launch):
    groups=event_aligned_windows(metadata,lanes,horizon=64,offsets=(-1,0,1),maximum=128)
    result={}
    for offset,item in groups.items():
        pairs=item['starts'].to(launch.device)
        if not len(pairs):result[offset]={'count':0,'errors':None};continue
        t,l=pairs.unbind(1);state=data['state'][t,l];feet=data['feet'][t,l].float();present=data['present'][t,l]
        errors={}
        for step in range(64):
            state,feet,present=ball_transition(base,ball,state,data['targets'][t+step,l],data['previous_action'][t+step,l],
                feet,present,data['elapsed'][t+step,l],launch[l],action_model=sweep)
            if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite64-step heldout ball rollout')
            if step+1 in (1,4,16,32,64):
                true=data['next_state'][t+step,l]
                errors[str(step+1)]=dict(velocity_rmse=float((state[:,52:55]-true[:,52:55]).square().mean().sqrt()),
                    position_rmse=float((state[:,49:52]-true[:,49:52]).square().mean().sqrt()),
                    forward_sign_error=float(((state[:,52]>0)!=(true[:,52]>0)).float().mean()))
        result[offset]=dict(count=len(pairs),starts=pairs.tolist(),errors=errors)
    return result


def sensitivity(base,ball,sweep,data,pairs,launch):
    if not len(pairs):return dict(count=0,gradient_norm=None)
    t,l=pairs.to(launch.device).unbind(1)
    state=data['state'][t,l]
    target=data['targets'][t,l].detach().clone().requires_grad_(True)
    other=[data[k][t,l] for k in ('previous_action','feet','present','elapsed')]
    pred,_,_=ball_transition(base,ball,state,target,*other,launch[l],action_model=sweep)
    gradients=[torch.autograd.grad(pred[:,52+j].sum(),target,retain_graph=True,allow_unused=True)[0] for j in range(3)]
    g=torch.stack([target.new_zeros(target.shape) if v is None else v for v in gradients],1)[:,:,12:18]
    return dict(count=len(pairs),gradient_norm=float(g.norm()),
        mean_per_case_norm=float(g.flatten(1).norm(dim=-1).mean()),finite=bool(torch.isfinite(g).all()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('physical-checkpoint','ball-checkpoint','packed-data','split-manifest','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--updates',type=int,default=400)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--sampling-schema', choices=('legacy-v1','contact-neighborhood-v2'), default='contact-neighborhood-v2')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.updates<1 or a.batch<4 or a.batch%4:raise ValueError('Positive updates and batch multiple of four required')
    if a.sampling_schema=='contact-neighborhood-v2' and a.batch%8:
        raise ValueError('Contact neighborhood batch must be divisible by eight')
    a.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(20261117)
    started=time.perf_counter()
    device=torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    (a.output/'startup.json').write_text(json.dumps(dict(argv=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seed=20261117,device=str(device),updates=a.updates,batch=a.batch,actor_updates=0),indent=2)+'\n')
    print('ACTION_BALL_STAGE=LOAD',flush=True)
    physical_checkpoint=torch.load(a.physical_checkpoint,map_location='cpu',weights_only=True)
    if physical_checkpoint['schema']!=PhysicalWorldV2.schema:raise ValueError('Expected physical55 v2')
    source=physical_checkpoint['source_manifest']['task_contract']
    asset_sha=source['asset']['sha256']
    base=PhysicalWorldV2(asset_sha).to(device)
    base.load_state_dict(physical_checkpoint['model'],strict=True)
    legacy_checkpoint=torch.load(a.ball_checkpoint,map_location='cpu',weights_only=True)
    if legacy_checkpoint['schema']!=BallResidualV2.schema:raise ValueError('Expected ball residual v2')
    ball=BallResidualV2().to(device)
    ball.load_state_dict(legacy_checkpoint['model'],strict=True)
    base.eval().requires_grad_(False);ball.eval().requires_grad_(False)
    sweep=ActionConditionedBallV1().to(device)
    package=torch.load(a.packed_data,map_location='cpu',weights_only=True)
    manifest=package.get('source_manifest',package.get('manifest',{}))
    if manifest.get('task_contract')!=source:
        raise ValueError('Packed data task/asset/execution contract is missing or differs from physical model')
    metadata=package['tensors']
    required=('state','next_state','targets','previous_action','feet','present','elapsed',
              'event_state','next_event_state','episode','terminated','timeout')
    if any(name not in metadata for name in required):raise ValueError('Packed data missing transition field')
    split=json.loads(a.split_manifest.read_text());train=set(split['train_lanes']);val=set(split['validation_lanes'])
    lanes=metadata['state'].shape[1]
    if not train or not val or train&val or min(train|val)<0 or max(train|val)>=lanes:
        raise ValueError('Train/validation lanes must be disjoint and in range')
    suite=source['suite'];path=Path(suite['path'])
    path=path if path.is_absolute() else Path(__file__).resolve().parents[1]/path
    if sha256(path)!=suite['sha256']:raise ValueError('Suite hash mismatch')
    cases=json.loads(path.read_text())['cases']
    if len(cases)!=lanes:raise ValueError('Packed lane order must match case order')
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in cases],device=device)
    data={k:v.to(device) for k,v in metadata.items()}
    events=(data['next_event_state'][...,1]>data['event_state'][...,1]) | (
        data['next_event_state'][...,6]>data['event_state'][...,6])
    mask=torch.zeros(lanes,dtype=torch.bool,device=device);mask[list(train)]=True
    eligible=data['present'].bool()&mask[None]
    positive=(eligible&events).nonzero()
    # The native event edge occurs in the NEXT recorded transition. Its
    # previous transition is a hard negative; training on it prevents the
    # model from inventing a bounce one control tick too early.
    preceding=preimpact_hard_negatives(events,data['episode'],data['terminated'],data['timeout'])
    hard_negative=(eligible&preceding&~events).nonzero()
    negative=(eligible&~events&~preceding).nonzero()
    neighborhoods=contact_neighborhood_pools(events,data['episode'],data['terminated'],data['timeout'],eligible)
    if a.sampling_schema=='contact-neighborhood-v2' and any(len(v)<8 for v in neighborhoods.values()):
        raise ValueError('Need at least eight real transitions in each contact neighborhood pool')
    allocated_draws={k:0 for k in neighborhoods}
    if min(len(positive),len(negative),len(hard_negative))<8:
        raise ValueError('Need real racket contacts and pre-contact hard negatives')
    if not len(event_aligned_windows(metadata,val,horizon=64,offsets=(0,),maximum=1)['0']['starts']):
        raise ValueError('No heldout contact edge for candidate validation')
    print('ACTION_BALL_STAGE=INITIAL_VALIDATION',flush=True)
    initial=metrics(base,ball,None,data,metadata,val,launch)
    initial_ar=autoregressive_metrics(base,ball,None,data,metadata,val,launch)
    generator=torch.Generator(device=device).manual_seed(20261117)
    optimizer=torch.optim.Adam(sweep.head.parameters(),lr=2e-4)
    history=[]
    print('ACTION_BALL_STAGE=FIT',flush=True)
    for update in range(1,a.updates+1):
        def draw(pool,n):return pool[torch.randint(len(pool),(n,),device=device,generator=generator)]
        if a.sampling_schema=='legacy-v1':
            pairs=torch.cat((draw(positive,a.batch//2),draw(hard_negative,a.batch//4),draw(negative,a.batch//4)))
        else:
            quota=dict(contact=a.batch//2,pre_contact=a.batch//4,post_contact=a.batch//8,background=a.batch//8)
            pairs=torch.cat([draw(neighborhoods[k],n) for k,n in quota.items()])
            for k,n in quota.items():allocated_draws[k]+=n
        t,l=pairs.unbind(1)
        state=data['state'][t,l];targets=data['targets'][t,l]
        with torch.no_grad():
            physical,_,_=base(state,targets,data['previous_action'][t,l],
                data['feet'][t,l].float(),data['present'][t,l],data['elapsed'][t,l],launch[l])
            prior,_=base.prior(state,data['present'][t,l],data['elapsed'][t,l],launch[l])
            legacy,_=ball(ball_geometry_features(state,base.geometry),data['present'][t,l])
            baseline=prior[:,49:]+legacy
        correction=sweep(state,physical,prior[:,49:],base.geometry,data['present'][t,l])
        predicted=baseline+correction
        true=data['next_state'][t,l,49:]
        scale=predicted.new_tensor([.1,.1,.1,2.,2.,2.])
        error=((predicted-true)/scale).square().mean(-1)
        loss=4.*error[:a.batch//2].mean()+error[a.batch//2:].mean()+.02*correction[a.batch//2:].square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite action-ball fitting loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(sweep.head.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        if update==1 or update%25==0 or update==a.updates:
            record=dict(update=update,loss=float(loss.detach()),gradient=float(gradient.detach()),elapsed_s=time.perf_counter()-started)
            history.append(record)
            (a.output/'progress.json').write_text(json.dumps(record)+'\n')
            print('ACTION_BALL_UPDATE='+json.dumps(record),flush=True)
    print('ACTION_BALL_STAGE=FINAL_VALIDATION',flush=True)
    final=metrics(base,ball,sweep,data,metadata,val,launch)
    candidate_ar=autoregressive_metrics(base,ball,sweep,data,metadata,val,launch)
    pairs=event_aligned_windows(metadata,val,horizon=64,offsets=(0,),maximum=16)['0']['starts']
    gradient=sensitivity(base,ball,sweep,data,pairs,launch)
    torch.save(dict(schema=sweep.schema,model=sweep.state_dict(),asset_sha256=asset_sha,
        physical_sha256=sha256(a.physical_checkpoint),legacy_ball_sha256=sha256(a.ball_checkpoint),
        packed_sha256=sha256(a.packed_data),split_sha256=sha256(a.split_manifest),
        geometry_hashes=sweep.preview.geometry_hashes,softness_m=sweep.softness_m,
        updates=a.updates,evidence_scope='recorded-command model-only, no native paired perturbations'),
        a.output/'action_ball_candidate.pt')
    comparison=dict(elapsed_s=time.perf_counter()-started,gpu_peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,initial=initial,candidate=final,initial_autoregressive=initial_ar,candidate_autoregressive=candidate_ar,action_gradient=gradient,history=history,
        action_ball_optimizer_updates=a.updates,base_world_updates=0,
        sampling_schema=a.sampling_schema,pool_sizes={k:len(v) for k,v in neighborhoods.items()},
        allocated_draws=allocated_draws if a.sampling_schema!='legacy-v1' else None,
        independence_scope="candidate fit heldout lanes; historical base-model holdout independence requires separate provenance",
        train_contact_transitions=int(len(positive)),train_noncontact_transitions=int(len(negative)),
        hard_precontact_transitions=int(len(hard_negative)),
        training_lanes=sorted(train),validation_lanes=sorted(val),native_paired_action_probe=False,
        actor_updates=0,world_updates=0,critic_updates=0,auto_promoted=False,
        candidate_sha256=sha256(a.output/'action_ball_candidate.pt'))
    (a.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(dict(candidate=str(a.output/'action_ball_candidate.pt'),comparison=comparison)))


if __name__=='__main__':main()
