#!/usr/bin/env python3
"""Fit the existing joint/root pose readout on native FK centers and paired executed-command prefixes."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_supermimic_contact_rollout import sha
from fit_supermimic_mixed_body_rollout import event_pools
from quadsmash.rl.supermimic_ball_residual_v2 import load_ball_residual
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state, robot_loss
from quadsmash.rl.supermimic_training_probe_replay import training_windows


def take_windows(data, ticks, lanes, horizon):
    index = ticks[None] + torch.arange(horizon)[:, None]
    return {k:data[k][index, lanes] for k in ('state','feet','present','elapsed','targets','previous_action','next_state')}


def pose_objective(world, ball, window, launches, pair_count, center_scales):
    state, feet, present = (window[k][0] for k in ('state','feet','present'))
    total = state.new_zeros(())
    diagnostics = {}
    for t in range(len(window['targets'])):
        state, feet, present = ball_transition(world, ball, state, window['targets'][t],
            window['previous_action'][t], feet, present, window['elapsed'][t], launches)
        # Preserve ball predictions as forward context without moving the robot through a ball-error gradient.
        state = torch.cat((state[:, :49], state[:, 49:].detach()), 1)
        if t+1 not in (1,2,4,8,16,32,64):
            continue
        truth = window['next_state'][t]
        pred = racket_state(state, world.geometry)[:, :3]
        actual = racket_state(truth, world.geometry)[:, :3]
        center = F.smooth_l1_loss((pred-actual)/.03, torch.zeros_like(pred))
        body = robot_loss(state, truth)
        # Native repeatability and full executed sequences are checked by training_windows().
        signed = state.new_zeros(())
        if pair_count:
            signs = pred[-2*pair_count:].reshape(2,pair_count,3)
            labels = actual[-2*pair_count:].reshape(2,pair_count,3)
            delta_error = (signs[1]-signs[0]) - (labels[1]-labels[0])
            signed = F.smooth_l1_loss(delta_error/center_scales[t], torch.zeros_like(delta_error))
        total = total + body + center + .25*signed
        diagnostics[t+1] = dict(robot=float(body.detach()), center=float(center.detach()), signed=float(signed.detach()))
    return total/len(diagnostics), diagnostics


@torch.no_grad()
def validate(world, ball, data, pairs, launches, device):
    """Frozen model, identical natural and contact windows, no threshold selection."""
    result = {}
    for subset, (ticks, lanes) in pairs.items():
        acc = {}
        for ts, ls in zip(ticks.split(16), lanes.split(16),strict=True):
            window = {k:v.to(device) for k,v in take_windows(data,ts,ls,64).items()}
            state, feet, present = (window[k][0] for k in ('state','feet','present'))
            for t in range(64):
                state,feet,present = ball_transition(world,ball,state,window['targets'][t],window['previous_action'][t],
                                                     feet,present,window['elapsed'][t],launches[ls].to(device))
                if t+1 not in (1,2,4,8,16,32,64):
                    continue
                truth = window['next_state'][t]
                pred = racket_state(state,world.geometry)
                real = racket_state(truth,world.geometry)
                metrics = dict(body_rmse=(state[:,:49]-truth[:,:49]).square().mean(1),
                               gyro_mae=(state[:,10:13]-truth[:,10:13]).abs().mean(1),
                               center_mae=(pred[:,:3]-real[:,:3]).abs().mean(1),
                               ball_velocity_rmse=(state[:,52:55]-truth[:,52:55]).square().mean(1))
                for k,v in metrics.items():
                    acc.setdefault((t+1,k),[]).append(v.cpu())
        result[subset] = {}
        for (h,k),v in acc.items():
            avg=torch.cat(v).mean()
            result[subset].setdefault(h,{})[k]=float(avg.sqrt() if k.endswith('rmse') else avg)
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('body','ball','replay','stage','packed','training-collection','output'):
        cli.add_argument('--'+name,type=Path,required=True)
    cli.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    cli.add_argument('--paired-development',type=Path,help='Fit shared Body layers on absolute and signed trajectories and select on independent cases')
    cli.add_argument('--long-body-anchor', action='store_true', help='Add measured 32/64-step Body and racket targets to the paired shared-layer fit')
    cli.add_argument('--arm-pose-velocity-readout',action='store_true',
                     help='Restrict paired Body updates to the six arm position and velocity output rows each')
    cli.add_argument('--signed-arm-supervision',action='store_true',
                     help='Share the signed supervision weight between racket FK and measured arm joint responses')
    cli.add_argument('--audit-arm-drive',action='store_true',help='Evaluate a train-only arm velocity calibration on short pairs and long native windows')
    cli.add_argument('--audit-center-timing',action='store_true',help='Compare fixed timing offsets on signed Body trajectories with no optimizer updates')
    cli.add_argument('--additional-training',type=Path,help='Use another independently scored native training launch panel')
    cli.add_argument('--additional-support',type=Path)
    a=cli.parse_args()
    if bool(a.additional_training)!=bool(a.additional_support) or (a.additional_training and not a.paired_development):
        cli.error('Additional pairs require a scored native collection and a paired development panel')
    if a.audit_arm_drive and (a.audit_center_timing or a.signed_arm_supervision or a.arm_pose_velocity_readout):
        cli.error('Arm drive identification is a separate read-only evaluation')
    if (a.long_body_anchor or a.audit_center_timing or a.audit_arm_drive) and not a.paired_development:
        cli.error('The shared Body calibration and timing diagnostic require a paired development panel')
    if a.arm_pose_velocity_readout and (not a.paired_development or a.audit_center_timing):
        cli.error('Arm readout updates require a paired development fit')
    if a.signed_arm_supervision and not a.arm_pose_velocity_readout:
        cli.error('Direct arm response fitting requires the arm pose and velocity readout')
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.manual_seed(20261121)
    torch.set_num_threads(3)
    if a.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.18 if a.paired_development else .09)
    parent=torch.load(a.body,map_location='cpu',mmap=True,weights_only=True)
    parent_hash=sha(a.body)
    task=parent['source_manifest']['task_contract']
    stage=torch.load(a.stage,map_location='cpu',mmap=True,weights_only=False)
    fit=json.loads((a.replay/'fit.json').read_text())
    base=stage['metadata']['physical_imagination']['checkpoint_sha256']
    if (sha(a.stage)!=fit['stage_checkpoint_sha256'] or base['physical']!=parent_hash or base['ball']!=sha(a.ball)
            or sha(a.packed)!=fit['packed_sha256'] or fit['packed_sha256']!=parent['calibration']['packed_sha256']):
        raise ValueError('Physical source, replay boundary or native data differs')
    replay=torch.load(a.replay/'online_world.pt',map_location='cpu',mmap=True,weights_only=True)
    if tensor_digest(replay.items())!=fit['online_world_sha256']:
        raise ValueError('Updated WM differs from the replayed checkpoint')
    world=PhysicalWorldV2(task['asset']['sha256']).to(a.device).eval()
    ball=load_ball_residual(torch.load(a.ball,map_location='cpu',mmap=True,weights_only=True)).to(a.device).eval()
    for prefix,model in [('physical_model.',world),('ball_model.',ball)]:
        model.load_state_dict({k.removeprefix(prefix):v for k,v in replay.items() if k.startswith(prefix)},strict=True)
        model.requires_grad_(False)
    layer=world.body.net[-1]
    layer.requires_grad_(True)
    mask=torch.zeros_like(layer.bias)
    mask[:7]=1; mask[25:31]=1
    initial=[v.detach().clone() for v in layer.parameters()]
    packed=torch.load(a.packed,map_location='cpu',mmap=True,weights_only=True)
    if packed['source_manifest']['task_contract']!=task:
        raise ValueError('Mismatch in physical setup')
    data=packed['tensors']
    suite=Path(task['suite']['path'])
    if sha(suite)!=task['suite']['sha256']:
        raise ValueError('Native suite changed')
    launches=torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps'] for c in json.loads(suite.read_text())['cases']])
    eligible=torch.zeros(len(launches),dtype=torch.bool)
    eligible[parent['calibration']['train_lanes']]=True
    eligible[fit['excluded_lanes']]=False
    excluded=(~eligible).nonzero().flatten().tolist()
    train,receipt=training_windows(a.training_collection,task,contract(),excluded)
    if len(train)!=1:
        raise ValueError('Expected one registered 16-step excitation specification')
    if a.paired_development:
        from supermimic_paired_body_candidate import fit_candidate
        fit_candidate(a,world,ball,data,launches,eligible,train,receipt,parent,fit,task,contract())
        return
    lanes,signed=train[0]
    if len(lanes)<2:
        raise ValueError('No independent signed training lanes')
    with torch.no_grad():
        labels = [racket_state(signed[b]['next_state'].flatten(0,1).to(a.device),world.geometry)[:,:3]
                  .reshape(16,len(lanes),3) for b in ('minus','plus')]
        scales=(labels[1]-labels[0]).square().mean((1,2)).sqrt().clamp_min(1e-5)
    horizons=(2,4,8,16)
    pools={h:event_pools(data,eligible,h,False) for h in horizons}
    held=event_pools(data,~eligible,64,False)
    rng=torch.Generator().manual_seed(20261121)
    val={k:tuple(rows[torch.randperm(len(rows),generator=rng)[:64]].T) for k,rows in held.items()}
    a.output.mkdir(parents=True)
    start=validate(world,ball,data,val,launches,a.device)
    opt=torch.optim.Adam(layer.parameters(),lr=1e-5)
    with (a.output/'training.jsonl').open('x',buffering=1) as log:
        for step,h in enumerate(horizons*4):
            pool=pools[h]
            rows=torch.cat([pool[k][torch.randperm(len(pool[k]),generator=rng)[:count]] for k,count in [('natural',10),('racket',2)]])
            natural=take_windows(data,*rows.T,h)
            ix=torch.randperm(len(lanes),generator=rng)[:2]
            window={k:torch.cat([v,*[signed[b][k][:h,ix] for b in ('minus','plus')]],1).to(a.device) for k,v in natural.items()}
            ids=torch.cat([rows[:,1],torch.tensor(lanes)[ix].repeat(2)])
            loss,terms=pose_objective(world,ball,window,launches[ids].to(a.device),len(ix),scales)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            layer.weight.grad.mul_(mask[:,None]);layer.bias.grad.mul_(mask)
            norm=torch.nn.utils.clip_grad_norm_(layer.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            log.write(json.dumps(dict(step=step+1,horizon=h,loss=float(loss.detach()),grad=float(norm),terms=terms),allow_nan=False)+'\n')
    for param,old in zip(layer.parameters(),initial):
        torch.testing.assert_close(param[mask==0],old[mask==0],atol=0,rtol=0)
    end=validate(world,ball,data,val,launches,a.device)
    metadata=dict(source_body_sha256=parent_hash,source_stage_sha256=sha(a.stage),source_world_sha256=fit['online_world_sha256'],
                  packed_sha256=fit['packed_sha256'],training=receipt,updates=16,device=a.device,
                  source_counts_per_batch=dict(natural=10,contact=2,signed=4),paired_scales_m=scales.tolist(),
                  validation_windows={k:torch.stack(v,1).tolist() for k,v in val.items()},before=start,after=end,
                  readout_rows=mask.nonzero().flatten().tolist(),actor_updates=0,policy_training_admitted=False)
    parent.update(model=world.cpu().state_dict(),calibration=dict(parent['calibration'],paired_center=metadata,
                  train_lanes=eligible.nonzero().flatten().tolist(),excluded_lanes=excluded),policy_training_admitted=False)
    atomic_torch_save(parent,a.output/'world.pt')
    metadata['model_sha256']=sha(a.output/'world.pt')
    (a.output/'fit.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
