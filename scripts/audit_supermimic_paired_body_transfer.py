#!/usr/bin/env python3
"""Measure native action-to-racket response by control step under the same nominal ball trajectory."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.functional import cosine_similarity

from quadsmash.rl.supermimic_physical_world_v1 import PhysicalWorldV1
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_toi_racket_objective import racket_features


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):
            h.update(part)
    return h.hexdigest()


def response_metrics(native,predicted,repeat):
    amplitude=native.norm(dim=-1)
    supported=amplitude>torch.maximum(amplitude.new_full(amplitude.shape,1e-8),3*repeat.norm(dim=-1))
    result=dict(rows=len(native),supported=int(supported.sum()),absolute_rmse=float((predicted-native).square().mean().sqrt()))
    if supported.any():
        gain=predicted[supported].norm(dim=-1)/amplitude[supported]
        cos=cosine_similarity(predicted[supported],native[supported],dim=-1)
        result.update(gain_median=float(gain.median()),gain_p10=float(gain.quantile(.1)),
                      cosine_median=float(cos.median()),cosine_p10=float(cos.quantile(.1)))
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('dataset','natural','candidate','suite','output'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--body-candidate',type=Path,help='Optional mixed-fit Body WM; the probe lanes must be held out')
    args=p.parse_args();torch.set_num_threads(3)
    ck=torch.load(args.candidate,map_location='cpu',weights_only=False,mmap=True)
    task=ck['metadata']['task_contract']
    if file_hash(args.suite)!=task['suite']['sha256']:raise ValueError('Different launch suite')
    natural=torch.load(args.natural,map_location='cpu',mmap=True,weights_only=False)
    if natural['schema']!=PhysicalWorldV1.schema:raise ValueError('Unknown natural model')
    v1=PhysicalWorldV1().eval();v1.load_state_dict(natural['model'],strict=True)
    body=PhysicalWorldV2(task['asset']['sha256']).eval()
    body.load_state_dict({k.removeprefix('physical_model.'):v for k,v in ck['world'].items() if k.startswith('physical_model.')},strict=True)
    dataset=torch.load(args.dataset,map_location='cpu',weights_only=True)
    if dataset['schema']!='supermimic-action-identification-v1':raise ValueError('Wrong native pair schema')
    rows=dataset['rows'];lanes=[r['lane_index'] for r in rows]
    launch=torch.tensor([r['launch']['position_m']+r['launch']['velocity_mps'] for r in json.loads(args.suite.read_text())['cases']])[lanes]
    def field(key):return torch.stack([r[key] for r in rows])
    def fk(s):return racket_features(s,body.geometry)
    nominal=field('step_next_states_nominal_a')
    native={name:field('step_next_states_'+name) for name in ('minus','plus','nominal_b')}
    spans=dict(position=slice(0,3),velocity=slice(3,6),normal=slice(6,9),angular_velocity=slice(9,12))
    result={}
    models = [('natural',v1),('candidate',body)]
    if args.body_candidate:
        fitted = torch.load(args.body_candidate,map_location='cpu',weights_only=True)
        if (set(fitted['calibration']['train_lanes']) & set(lanes) or fitted['source_manifest']['task_contract']!=task):
            raise ValueError('Mixed-fit body has incompatible task or overlapping action-ID lanes')
        mixed = PhysicalWorldV2(task['asset']['sha256']).eval()
        mixed.load_state_dict(fitted['model'],strict=True)
        models.append(('mixed_body',mixed))
    for name,model in models:
        pred={}
        for branch in ('minus','plus'):
            state=field('state');previous=field('previous_action');feet=field('foot_contact').float()
            elapsed=field('elapsed');present=field('present');targets=field('step_targets_'+branch);actions=field('step_final_actions_'+branch)
            pred[branch]=[]
            for i in range(targets.shape[1]):
                s,logits,present=model(state,targets[:,i],previous,feet,present,elapsed,launch)
                pred[branch].append(fk(s))
                state=torch.cat((s[:,:49],nominal[:,i,49:]),dim=1)
                feet=logits[:,:4].sigmoid();previous=actions[:,i];elapsed=elapsed+.01
        metrics={}
        for i in range(nominal.shape[1]):
            signal=fk(native['plus'][:,i])-fk(native['minus'][:,i])
            response=pred['plus'][i]-pred['minus'][i]
            noise=fk(native['nominal_b'][:,i])-fk(nominal[:,i])
            metrics[i+1]={key:response_metrics(signal[:,span],response[:,span],noise[:,span]) for key,span in spans.items()}
        result[name]=metrics
    payload=dict(models=result,source_sha256={k:file_hash(getattr(args,k)) for k in ('candidate','natural','dataset','suite')},
                 lanes=lanes,specs=[r['spec_name'] for r in rows],rows=len(rows),optimizer_updates=0,
                 scope='diagnostic held-command pairs; common recorded nominal ball between steps, measured branch commands, predicted robot/contact')
    if args.body_candidate: payload['source_sha256']['mixed_body'] = file_hash(args.body_candidate)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
