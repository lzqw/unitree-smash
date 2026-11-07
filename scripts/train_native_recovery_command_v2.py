#!/usr/bin/env python3
"""Event-weighted supervised refinement of the native recovery command head.

Only verified positive recovery episodes are used. Onset emphasis and a
worst-joint term address large event-local errors hidden by episode mean MSE.
This is supervised initialization, not PPO or a closed-loop success claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_native_recovery_command_v1 import load_episodes,evaluate_fit
from quadsmash.control.learned_recovery_command_v1 import RecoveryCommandHead,compose_native_delta


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def evaluate_events(head,episodes,limits):
    rows=evaluate_fit(head,episodes,limits);width=limits[:,1]-limits[:,0]
    for row,e in zip(rows,episodes):
        command=compose_native_delta(e['base'],head(e['features']),e['previous'],
            torch.ones(len(e['base']),dtype=torch.bool),limits)
        error=abs(command-e['target'])*width/2.
        row.update(onset_max_error_rad=float(error[0].max()),
            first_five_max_error_rad=float(error[:5].max()),
            first_second_max_error_rad=float(error[:100].max()))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--seed',type=int,default=882081)
    p.add_argument('--max-seconds',type=float,default=900.)
    a=p.parse_args()
    if a.output.exists() or a.steps<1 or a.max_seconds<=0:raise ValueError('Fresh output and finite training budget required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    manifest,episodes,limits=load_episodes(a.dataset)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['dataset_manifest_sha256']!=sha(a.dataset/'manifest.json'):
        raise ValueError('Refinement must use the original verified dataset')
    if checkpoint['base_policy_sha256']!=manifest['base_policy_sha256'] or checkpoint['protocol_sha256']!=manifest['protocol_sha256']:
        raise ValueError('Frozen base or task differs')
    head=RecoveryCommandHead.from_checkpoint(checkpoint).train()
    optimizer=torch.optim.Adam(head.parameters(),lr=3e-4)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    groups=[[i for i,e in enumerate(episodes) if e['corrected']==flag] for flag in (False,True)]
    if not all(groups):raise ValueError('Require correction and unchanged successful episodes')
    before=evaluate_events(head,episodes,limits);a.output.mkdir(parents=True)
    width=limits[:,1]-limits[:,0];start=time.monotonic();history=[]
    for step in range(a.steps):
        selected=[]
        for _ in range(512):
            e=episodes[int(rng.choice(groups[int(rng.integers(2))]))]
            # Equal exposure to first 50 ms, first second, and full recovery.
            length=[min(len(e['features']),5),min(len(e['features']),100),len(e['features'])][int(rng.integers(3))]
            selected.append((e,int(rng.integers(length))))
        features=torch.stack([e['features'][k] for e,k in selected])
        target=torch.stack([e['delta'][k] for e,k in selected])
        square=((head(features)-target)*width).square()
        loss=square.mean()+.25*square.max(dim=1).values.mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(gradient):raise ValueError('Nonfinite supervised update')
        optimizer.step()
        if step%250==0 or step+1==a.steps:
            row=dict(step=step+1,loss=float(loss.detach()),elapsed_s=time.monotonic()-start)
            history.append(row);print(json.dumps(row),flush=True)
        if time.monotonic()-start>=a.max_seconds:break
    after=evaluate_events(head.eval(),episodes,limits)
    checkpoint.update(head_state_dict=head.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        torch_rng_state=torch.get_rng_state(),numpy_rng_state=rng.bit_generator.state,
        steps=checkpoint['steps']+step+1,refinement_parent_sha256=sha(a.checkpoint),
        refinement='equal first5 / first100 / full recovery; balanced episodes; mean plus worst-joint loss')
    torch.save(checkpoint,a.output/'head.pt')
    report=dict(schema='native-recovery-command-event-refinement-v2',dataset=str(a.dataset.resolve()),
        seed=a.seed,refinement_steps=step+1,elapsed_s=time.monotonic()-start,
        parent_checkpoint_sha256=sha(a.checkpoint),checkpoint_sha256=sha(a.output/'head.pt'),
        before=before,after=after,history=history,
        native_evaluated=False,ppo_training=False,case_id_as_input=False,
        scope='In-sample command fitting only; original actor/normalizer/interface and task retained.',
        source_sha256={str(f.resolve()):sha(f) for f in [Path(__file__),a.checkpoint,a.dataset/'manifest.json',
            ROOT/'scripts/train_native_recovery_command_v1.py',ROOT/'quadsmash/control/learned_recovery_command_v1.py']})
    (a.output/'fit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('refinement_steps','elapsed_s','checkpoint_sha256')}))


if __name__=='__main__':main()
