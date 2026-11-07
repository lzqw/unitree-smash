#!/usr/bin/env python3
"""Audit the learned adapter over recorded observations, without physics replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.learned_recovery_command_v1 import (
    admit_native_recovery_head,NativeRecoveryCommandAdapter)
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.recovery_command_dataset_v1 import recovery_eligibility


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous audits')
    torch.set_num_threads(1)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['dataset_manifest_sha256']!=sha(args.dataset/'manifest.json'):
        raise ValueError('Checkpoint and dataset differ')
    limits=np.asarray(manifest['joint_limits_rad']);width=limits[:,1]-limits[:,0]
    head=admit_native_recovery_head(checkpoint,manifest['base_policy_sha256'],
        manifest['protocol_sha256'],limits,.01)
    episodes=[]
    for entry in manifest['episodes']:
        path=args.dataset/entry['data_file']
        if sha(path)!=entry['data_sha256']:raise ValueError('Changed episode')
        with np.load(path,allow_pickle=False) as data:
            public=unpack_public(data['packed_public'])
            active,_=recovery_eligibility(public)
            base=data['native_action'].copy()
            known=data['student_request_known']
            base[known]=2.*(data['student_requested_joint_target_rad'][known]-limits[:,0])/width-1.
        episodes.append((public,base,active))
    obs=torch.tensor(np.stack([e[0] for e in episodes]),dtype=torch.float32)
    base=torch.tensor(np.stack([e[1] for e in episodes]),dtype=torch.float32)
    expected=torch.tensor(np.stack([e[2] for e in episodes]))
    adapter=NativeRecoveryCommandAdapter(head,len(episodes),'cpu',limits)
    previous=None;max_slew_violation=0.;active_ticks=0
    for tick in range(obs.shape[1]):
        command=adapter.action(obs[:,tick],base[:,tick])
        if not torch.equal(adapter.active,expected[:,tick]):raise ValueError('Causal gate mismatch')
        if not torch.equal(command[~adapter.active],base[:,tick][~adapter.active]):
            raise ValueError('Pre-recovery base command changed')
        if not torch.isfinite(command).all() or abs(command).max()>1.+1e-6:
            raise ValueError('Invalid native output')
        if previous is not None and adapter.active.any():
            step=abs(command[adapter.active,12:]-previous[adapter.active,12:])*torch.tensor(width[12:])/2.
            violation=float((step-.1).clamp_min(0.).max())
            max_slew_violation=max(max_slew_violation,violation)
            if violation>1e-6:raise ValueError('Actual-command arm slew violated')
        active_ticks+=int(adapter.active.sum());previous=command.clone()
    result=dict(schema='learned-recovery-command-adapter-audit-v1',passed=True,
        episodes=len(episodes),ticks=int(obs.shape[1]),active_ticks=active_ticks,
        nonrecovery_command_error_max=0.,arm_target_slew_violation_max_rad=max_slew_violation,
        scope='recorded observations with autoregressive command history; no simulated state transitions',
        native_execution=False,teacher_off_success_measured=False,
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),args.checkpoint,
            args.dataset/'manifest.json',ROOT/'quadsmash/control/learned_recovery_command_v1.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
