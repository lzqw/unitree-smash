#!/usr/bin/env python3
"""Calibrate the three root angular velocity outputs of a mixed-fit body on native training lanes."""
import argparse
import json
from pathlib import Path

import torch

from fit_supermimic_mixed_body_rollout import digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_policy_execution_v1 import contract
from quadsmash.rl.supermimic_training_probe_replay import training_windows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('body','packed','training-collection','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--ridge', default=.1, type=float)
    args = parser.parse_args()
    if not 0 < args.ridge < float('inf'):
        parser.error('ridge must be finite and positive')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    source = torch.load(args.body, weights_only=True, map_location='cpu')
    meta = source['calibration']
    task = source['source_manifest']['task_contract']
    if digest(args.packed) != meta['packed_sha256']:
        raise ValueError('Different training archive')
    packed = torch.load(args.packed, weights_only=True, mmap=True, map_location='cpu')
    if packed['source_manifest']['task_contract'] != task:
        raise ValueError('Different native task')
    data = packed['tensors']
    suite = Path(task['suite']['path'])
    if digest(suite) != task['suite']['sha256']:
        raise ValueError('Different launch suite')
    launch = torch.tensor([r['launch']['position_m']+r['launch']['velocity_mps']
                           for r in json.loads(suite.read_text())['cases']])
    allowed = set(meta['train_lanes'])
    if allowed & set(meta['excluded_lanes']):
        raise ValueError('Training and held-out lanes overlap')
    windows, receipt = training_windows(args.training_collection, task, contract(), set(range(len(launch)))-allowed)
    if receipt not in meta.get('sequence_receipts', [meta.get('sequence_receipt')]):
        raise ValueError('Signed training windows differ from the parent fit')
    model = PhysicalWorldV2(task['asset']['sha256']).eval().requires_grad_(False).to(device)
    model.load_state_dict(source['model'], strict=True)
    layer = model.body.net[-1]
    trunk = model.body.net[:-1]
    width = layer.in_features+1
    # Each axis gets a single Huber reweighting of the parent errors in physical units.
    gram = torch.zeros(3,width,width,dtype=torch.float64)
    rhs = torch.zeros(3,width,dtype=torch.float64)
    components = slice(10,13)
    keys = ('state','targets','previous_action','feet','present','elapsed')
    def accumulate(tensors, ticks, columns, lane_ids, weight):
        for t, c, lane in zip(ticks.split(512),columns.split(512),lane_ids.split(512)):
            inp = [tensors[k][t,c].to(device) for k in keys]
            features = model.features(*inp,launch[lane].to(device))
            hidden = trunk((features-model.input_mean)/model.input_scale)
            x = torch.cat([hidden,torch.ones_like(hidden[:,:1])],dim=-1)
            scale = model.delta_scale[components]
            pred = inp[0][:,components]+model.delta_mean[components]+scale*layer(hidden)[:,components]
            error = tensors['next_state'][t,c,components].to(device)-pred
            huber = (.1/error.abs().clamp_min(.1)).T.contiguous()
            for i in range(3):
                wx = huber[i,:,None]*x
                gram[i].add_((wx.T@x).cpu().double(), alpha=weight)
                rhs[i].add_((wx.T@(error[:,i]/scale[i])).cpu().double(), alpha=weight)
    natural = ~(data['terminated'] | data['timeout'])
    valid_lanes = torch.zeros(natural.shape[1],dtype=torch.bool)
    valid_lanes[list(allowed)] = True
    ticks, lanes = torch.where(natural & valid_lanes)
    accumulate(data,ticks,lanes,lanes,.75/len(ticks))
    signed_rows = sum(t['state'].shape[0]*t['state'].shape[1] for _,s in windows for t in s.values())
    for lane_ids, signed in windows:
        for t in signed.values():
            tick, col = torch.where(torch.ones_like(t['terminated'],dtype=torch.bool))
            accumulate(t,tick,col,torch.tensor(lane_ids)[col],.25/signed_rows)
    system = ((gram+gram.transpose(-1,-2))*.5+args.ridge*torch.eye(width,dtype=gram.dtype)).to(device)
    if not torch.isfinite(system).all() or not torch.isfinite(rhs).all():
        raise ValueError('Nonfinite native training features or targets')
    factor = torch.linalg.cholesky(system)
    solution = torch.cholesky_solve(rhs.to(device).unsqueeze(-1),factor).squeeze(-1)
    if not torch.isfinite(solution).all():
        raise ValueError('Nonfinite angular velocity readout')
    layer.weight[components].add_(solution[:,:-1].to(layer.weight))
    layer.bias[components].add_(solution[:,-1].to(layer.bias))
    info = dict(parent_sha256=digest(args.body),signed_rows=signed_rows,natural_rows=len(ticks),
                signed_weight=.25, ridge=args.ridge,huber_beta_rad_s=.1,
                readout_rows=[10,11,12],actor_updates=0,critic_updates=0)
    source.update(model=model.cpu().state_dict(),calibration=dict(meta,gyro_readout=info),policy_training_admitted=False)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.save(source,args.output/'world.pending')
    (args.output/'world.pending').replace(args.output/'world.pt')
    (args.output/'fit.json').write_text(json.dumps(dict(info,sha256=digest(args.output/'world.pt')),indent=2)+'\n')


if __name__ == '__main__':
    main()
