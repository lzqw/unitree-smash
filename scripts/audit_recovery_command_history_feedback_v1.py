#!/usr/bin/env python3
"""Isolate learned command-history feedback on fixed demonstration observations.

Physical observations and frozen-base requests remain recorded teacher values.
Only the explicitly supplied previous executed command is recursively replaced
by the learned output. This is an input-feedback diagnostic, NOT a simulation,
successful recovery label, or causal explanation of all native divergence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_native_recovery_command_v1 import load_episodes
from quadsmash.control.learned_recovery_command_v1 import RecoveryCommandHead,compose_native_delta


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def inspect_episode(head,episode,limits):
    x=episode['features'];base=episode['base'];target=episode['target'];width=limits[:,1]-limits[:,0]
    active=torch.ones(len(x),dtype=torch.bool)
    forced=compose_native_delta(base,head(x),episode['previous'],active,limits)
    previous=episode['previous'][:1].clone();recursive=[]
    for tick in range(len(x)):
        features=torch.cat((x[tick:tick+1,:812],previous,x[tick:tick+1,830:]),dim=1)
        previous=compose_native_delta(base[tick:tick+1],head(features),previous,active[:1],limits)
        recursive.append(previous[0])
    recursive=torch.stack(recursive)
    def metrics(command):
        error=abs(command-target)*width/2.
        return dict(rmse_rad=float(error.square().mean().sqrt()),max_rad=float(error.max()),
                    first_second_rmse_rad=float(error[:100].square().mean().sqrt()),
                    first_second_max_rad=float(error[:100].max()),onset_max_rad=float(error[0].max()))
    return dict(case=episode['case'],corrected=episode['corrected'],outcome=episode['outcome'],
        samples=len(x),teacher_history=metrics(forced),recursive_command_history=metrics(recursive),
        recursive_vs_forced_max_rad=float((abs(recursive-forced)*width/2.).max()),
        original_physical_observations_retained=True,native_rollout=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous evidence')
    torch.set_num_threads(1)
    manifest,episodes,limits=load_episodes(a.dataset)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['dataset_manifest_sha256']!=sha(a.dataset/'manifest.json'):
        raise ValueError('Use the fitted verified dataset')
    head=RecoveryCommandHead.from_checkpoint(checkpoint)
    rows=[inspect_episode(head,e,limits) for e in episodes]
    groups={}
    for name,flag in [('unchanged',False),('corrected',True)]:
        selected=[r for r in rows if r['corrected']==flag]
        groups[name]=dict(episodes=len(selected),
            teacher_history_mean_episode_rmse_rad=float(np.mean([r['teacher_history']['rmse_rad'] for r in selected])),
            recursive_mean_episode_rmse_rad=float(np.mean([r['recursive_command_history']['rmse_rad'] for r in selected])),
            recursive_max_rad=max(r['recursive_command_history']['max_rad'] for r in selected))
    sources=[Path(__file__),a.checkpoint,a.dataset/'manifest.json',ROOT/'scripts/train_native_recovery_command_v1.py',
        ROOT/'quadsmash/control/learned_recovery_command_v1.py']
    report=dict(schema='recovery-command-history-feedback-v1',scope=__doc__,groups=groups,cases=rows,
        source_sha256={str(f.resolve()):sha(f) for f in sources},native_execution=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(groups))


if __name__=='__main__':main()
