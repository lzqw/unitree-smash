#!/usr/bin/env python3
"""Refit the existing robot predictor with natural rollouts and native paired racket responses."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_native_response_data_v1 import load_response_data, normalized_response_loss
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state, racket_loss, robot_loss
from quadsmash.rl.supermimic_toi_racket_objective import racket_features


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def rollout_terms(model,ball,data,launch,ticks,lanes,horizon):
    ids = (ticks,lanes)
    state,feet,present = (data[k][ids] for k in ('state','feet','present'))
    body = state.new_zeros(())
    racket = state.new_zeros(())
    for step in range(horizon):
        ix = (ticks+step,lanes)
        state,feet,present = ball_transition(model,ball,state,data['targets'][ix],data['previous_action'][ix],
                                             feet,present,data['elapsed'][ix],launch[lanes])
        truth = data['next_state'][ix]
        body = body + robot_loss(state,truth)
        racket = racket + racket_loss(racket_state(state,model.geometry), racket_state(truth,model.geometry))
    return body/horizon, racket/horizon


def pair_terms(model, pairs, index):
    args = [pairs[k][index] for k in ('previous_action','feet','present','elapsed','launch')]
    predictions, labels, point = [], [], 0.
    for name in ('minus','plus'):
        state = model(pairs['state'][index],pairs['target_'+name][index],*args,detach_contacts=True)[0]
        truth = pairs['next_'+name][index]
        point = point + robot_loss(state,truth)/2
        predictions.append(racket_features(state,model.geometry))
        labels.append(racket_features(truth,model.geometry))
    # Fixed physical training scales for p, v, n, omega. Compare response per executed radian.
    scale = predictions[0].new_tensor([.03]*3+[1.]*3+[1.]*3+[3.]*3)
    jac, support = normalized_response_loss(*predictions,*labels,pairs['target_minus'][index],
                                            pairs['target_plus'][index],scale)
    return point,jac,support



def sequence_terms(model, ball, signed, index, launch, horizon):
    """Differentiate the response to the full measured paired command prefix."""
    feature_scale = launch.new_tensor([.03]*3+[1.]*3+[1.]*3+[3.]*3)
    point = launch.new_zeros(())
    jacobian = launch.new_zeros(())
    stats = {}
    rollout = {}
    for sign in ('minus','plus'):
        d = signed[sign]
        rollout[sign] = (d['state'][0,index], d['feet'][0,index], d['present'][0,index])
    for t in range(horizon):
        pred, actual = [], []
        for sign in ('minus','plus'):
            d = signed[sign]
            state, feet, present = rollout[sign]
            rollout[sign] = ball_transition(model, ball, state, d['targets'][t,index],
                                            d['previous_action'][t,index], feet, present,
                                            d['elapsed'][t,index], launch[index])
            state = rollout[sign][0]
            truth = d['next_state'][t,index]
            point = point + robot_loss(state,truth)/(2*horizon)
            pred.append(racket_features(state,model.geometry))
            actual.append(racket_features(truth,model.geometry))
        # Prefix norm keeps the response identifiable after the held targets reconverge.
        targets = [signed[sign]['targets'][:t+1,index].transpose(0,1).flatten(1)
                   for sign in ('minus','plus')]
        loss, support = normalized_response_loss(*pred,*actual,*targets,feature_scale)
        jacobian = jacobian + loss/horizon
        if t+1 in (1,2,4,8,16):
            stats[f'tick_{t+1}'] = dict(**support, jacobian_loss=float(loss.detach()))
    return point,jacobian,stats

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source','ball','packed','split','reference-verification','probe-holdout','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--training-collection',type=Path,
                     help='Use verified held-action windows for half of identification pairs in each batch')
    cli.add_argument('--batch',type=int,default=32)
    cli.add_argument('--updates-per-horizon',type=int,default=8)
    cli.add_argument('--lr',type=float,default=1e-5)
    a=cli.parse_args()
    if a.batch<8 or a.batch%8 or a.updates_per_horizon<1 or not 0<a.lr<1:
        cli.error('Expected positive learning rate, steps and a batch divisible by eight')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU calibration requested but CUDA is unavailable')
    device=torch.device('cuda:0');torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(20261121)
    parent_cfg=json.loads((a.source.parent/'config.json').read_text())
    packed_hash=digest(a.packed)
    if parent_cfg['source_sha256']['packed']!=packed_hash:
        raise ValueError('Different physical archive than the base fit')
    source=torch.load(a.source,map_location='cpu',weights_only=True,mmap=True)
    task=source['source_manifest']['task_contract']
    data_file=torch.load(a.packed,weights_only=True,map_location='cpu',mmap=True)
    data=data_file['tensors']
    if data_file['source_manifest']['task_contract']!=task:
        raise ValueError('Native archive task contract mismatch')
    world=PhysicalWorldV2(task['asset']['sha256']).to(device).eval()
    world.load_state_dict(source['model'])
    world.requires_grad_(False);world.body.requires_grad_(True)
    ball=BallResidualV2().to(device).eval()
    ball_ck=torch.load(a.ball,map_location='cpu',weights_only=True)
    ball.load_state_dict(ball_ck['model']);ball.requires_grad_(False)
    suite=Path(task['suite']['path'])
    if digest(suite)!=task['suite']['sha256']:
        raise ValueError('Launch suite mismatch')
    launch=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps']
                         for c in json.loads(suite.read_text())['cases']],device=device)
    held=(set(parent_cfg['provenance']['validation_lanes'])
          |set(json.loads(a.reference_verification.read_text())['excluded_lanes'])
          |set(json.loads(a.probe_holdout.read_text())['validation_lanes']))
    lanes=sorted(set(parent_cfg['provenance']['train_lanes'])-held)
    pair_sources=[]
    for source_row in parent_cfg['native_response_sources']:
        values,meta=load_response_data(source_row['path'],a.split)
        if (meta['source_sha256']!=source_row['provenance']['source_sha256'] or meta['task_contract']!=task or
                not set(lanes)<=set(meta['train_lanes'])):
            raise ValueError('Missing training split or measured paired source changed')
        pair_sources.append({k:v[lanes].to(device) for k,v in values.items()})
    pairs={k:torch.cat([r[k] for r in pair_sources]) for k in pair_sources[0]}
    pair_count=len(pairs['state'])
    count_pair=a.batch//4
    count_event=a.batch//8
    count_natural=a.batch-count_pair-count_event
    if not lanes or pair_count<count_pair:
        raise ValueError('Not enough distinct measured training pairs')
    sequence = None
    sequence_receipt = None
    if a.training_collection:
        from quadsmash.rl.supermimic_policy_execution_v1 import contract
        from quadsmash.rl.supermimic_training_probe_replay import training_windows
        windows, sequence_receipt = training_windows(a.training_collection, task, contract(),
                                                    set(range(len(launch)))-set(lanes))
        sequence_lanes = [lane for ids,_ in windows for lane in ids]
        if len(sequence_lanes)<count_pair//2:
            raise ValueError('Too few eligible signed training windows')
        needed = ('state','feet','present','targets','previous_action','elapsed','next_state')
        sequence = {b:{k:torch.cat([signed[b][k] for _,signed in windows],dim=1).to(device)
                       for k in needed} for b in ('minus','plus')}
        sequence_launch = launch[sequence_lanes]
    horizons=(1,2,4,8,16,32,64)
    pools={}
    contact=data['next_event_state'][...,1]>data['event_state'][...,1]
    for h in horizons:
        valid=valid_windows(data['episode'],data['terminated'],data['timeout'],h)
        eligible = torch.zeros(valid.shape[1], dtype=torch.bool)
        eligible[lanes] = True
        valid &= eligible
        event=contact.unfold(0,h,1).any(-1)
        pools[h]=((valid&~event).nonzero(),(valid&event).nonzero())
        if any(len(x)<n for x,n in zip(pools[h],(count_natural,count_event))):
            raise ValueError('Not enough reset-safe native windows')
    data={k:data[k].to(device) for k in ('state','next_state','targets','previous_action','feet','present','elapsed')}
    metadata=dict(source_sha256=digest(a.source),ball_sha256=digest(a.ball),packed_sha256=packed_hash,
                  train_lanes=lanes,excluded_lanes=sorted(held),pair_sources=[r['provenance']['source_sha256'] for r in parent_cfg['native_response_sources']],
                  pair_count=pair_count,batch=a.batch,natural_count=count_natural,event_count=count_event,
                  pair_count_per_update=count_pair,sequence_pairs_per_update=count_pair//2 if sequence else 0,
                  sequence_receipt=sequence_receipt,horizons=horizons,updates_per_horizon=a.updates_per_horizon,
                  seed=20261121,lr=a.lr,actor_updates=0,critic_updates=0,policy_training_admitted=False)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    opt=torch.optim.Adam(world.body.parameters(),lr=a.lr)
    initial={k:v.detach().clone() for k,v in world.body.named_parameters()}
    step=0
    with (a.output/'fit.jsonl').open('x',buffering=1) as f:
        for h in horizons:
            for _ in range(a.updates_per_horizon):
                step+=1
                # Without replacement inside each stratum in a batch.
                selected=torch.cat([pool[torch.randperm(len(pool),generator=generator)[:n]]
                                    for pool,n in zip(pools[h],(count_natural,count_event))]).to(device)
                body,fk=rollout_terms(world,ball,data,launch,*selected.T,h)
                n_seq = count_pair//2 if sequence else 0
                ids=torch.randperm(pair_count,generator=generator)[:count_pair-n_seq].to(device)
                point,jac,support=pair_terms(world,pairs,ids)
                measured_steps = 2*len(ids)
                if sequence:
                    seq_ids=torch.randperm(len(sequence_launch),generator=generator)[:n_seq].to(device)
                    seq_h=min(h,sequence['minus']['state'].shape[0])
                    seq_point,seq_jac,support['sequence_by_tick']=sequence_terms(
                        world,ball,sequence,seq_ids,sequence_launch,seq_h)
                    point=(len(ids)*point+n_seq*seq_point)/count_pair
                    jac=(len(ids)*jac+n_seq*seq_jac)/count_pair
                    measured_steps+=2*n_seq*seq_h
                anchor=sum((v-initial[k]).square().mean() for k,v in world.body.named_parameters())
                loss=.75*(body+fk)+.25*(point+jac)+.01*anchor
                opt.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(world.body.parameters(),5.,error_if_nonfinite=True)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite robot fit')
                opt.step()
                metrics=dict(update=step,horizon=h,natural_and_event_transitions=len(selected)*h,
                             signed_pair_transitions=measured_steps,body=float(body.detach()),racket=float(fk.detach()),
                             jacobian=float(jac.detach()),pair_point=float(point.detach()),norm=float(grad),**support)
                f.write(json.dumps(metrics,allow_nan=False)+'\n')
    payload=dict(schema=world.schema,model=world.state_dict(),source_manifest=source['source_manifest'],
                 calibration=metadata,world_updates=step,policy_training_admitted=False)
    torch.save(payload,a.output/'world.pending')
    (a.output/'world.pending').replace(a.output/'world.pt')
    (a.output/'completed.json').write_text(json.dumps(dict(updates=step,device=torch.cuda.get_device_name(),
        peak_bytes=torch.cuda.max_memory_allocated(),candidate_sha256=digest(a.output/'world.pt'),
        actor_updates=0,critic_updates=0,policy_training_admitted=False),indent=2)+'\n')


if __name__=='__main__':
    main()
