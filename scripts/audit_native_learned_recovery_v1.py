#!/usr/bin/env python3
"""Verify teacher-off native command execution against recorded policy inputs."""
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
from scripts.compare_frozen_complete5_v2 import read_run


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audits')
    torch.set_num_threads(1)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(ROOT)
    outcomes,_=read_run(a.run,protocol)
    inputs=json.loads((a.run/'INPUTS.json').read_text())
    job=json.loads((a.run/'job.json').read_text())
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Native job incomplete')
    if (inputs.get('landing_mpc_cases') is not None or inputs.get('reactive_vhsip_cases') is not None or
            inputs.get('expert_bank') is not None or inputs.get('extra_recovery_teacher_enabled') is not False):
        raise ValueError('Additional runtime teacher must be disabled')
    if sha(a.checkpoint)!=inputs['learned_recovery_command_head_sha256']:
        raise ValueError('Wrong learned checkpoint')
    source_hashes=json.loads((a.run/'execution_source_hashes.json').read_text())
    module='quadsmash/control/learned_recovery_command_v1.py'
    if sha(ROOT/module)!=source_hashes[module]:raise ValueError('Audit adapter differs from native source')
    native=json.loads((a.run/'native_command_inputs.json').read_text())
    limits=np.asarray(native['joint_limits_rad']);width=limits[:,1]-limits[:,0]
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    head=admit_native_recovery_head(checkpoint,inputs['policy_sha256'],inputs['protocol_sha256'],limits,.01)
    traces=a.run/'fixed_policy_traces';manifest=json.loads((traces/'manifest.json').read_text())
    history_path=a.run/'learned_recovery_commands.npz'
    with np.load(history_path,allow_pickle=False) as data:
        records={key:data[key].copy() for key in data.files}
    checks=[];hashes={}
    for entry in manifest['episodes']:
        lane=entry['env_id'];case=entry['source_training_index'];n=entry['samples']
        if case!=lane:raise ValueError('Unexpected TRAIN128 lane ordering')
        path=traces/entry['data_file']
        if sha(path)!=entry['data_sha256']:raise ValueError('Changed native trace')
        hashes[str(path.resolve())]=entry['data_sha256']
        with np.load(path,allow_pickle=False) as trace:
            public=unpack_public(trace['packed_public'])
            actual=trace['dispatched_actions'].copy()
        if hashlib.sha256(public.tobytes()).hexdigest()!=entry['full_public794_sha256']:
            raise ValueError('Changed actual observations')
        if not records['first_episode'][:n,lane].all() or records['first_episode'][n:,lane].any():
            raise ValueError('First episode recording mask differs')
        if not np.array_equal(actual,records['final_command'][:n,lane]):
            raise ValueError('Head telemetry differs from actual dispatched action')
        adapter=NativeRecoveryCommandAdapter(head,1,'cpu',limits)
        replay_max=0.;delta_max=0.;slew_max=0.;active_ticks=0
        for tick in range(n):
            base=records['base_command'][tick,lane:lane+1]
            # Replay the actual per-tick input history, rather than accumulate
            # CPU/GPU rounding into a different counterfactual command history.
            if tick:
                adapter.previous.copy_(torch.from_numpy(actual[tick-1:tick]))
            replay=adapter.action(torch.from_numpy(public[tick:tick+1]),torch.from_numpy(base)).numpy()[0]
            active=bool(adapter.active[0])
            if active!=bool(records['active'][tick,lane]):raise ValueError('Causal gate differs')
            if not active and not np.array_equal(actual[tick],base[0]):
                raise ValueError('Nonrecovery command changed')
            replay_max=max(replay_max,float(np.max(abs(replay-actual[tick])*width/2.)))
            delta_max=max(delta_max,float(np.max(abs(adapter.last_delta.numpy()[0]-records['learned_delta'][tick,lane]))))
            if active:
                if tick==0:raise ValueError('No ground command history at takeover')
                slew_max=max(slew_max,float(np.maximum(0.,abs(actual[tick,12:]-actual[tick-1,12:])*width[12:]/2.-.1).max()))
            active_ticks+=active
        if replay_max>5e-5 or delta_max>2e-5 or slew_max>5e-6:
            raise ValueError(f'Native head mapping mismatch in case {case}: {replay_max}, {delta_max}, {slew_max}')
        checks.append(dict(case=case,samples=n,active_ticks=active_ticks,
            cpu_vs_native_command_max_rad=replay_max,cpu_vs_native_delta_max=delta_max,
            native_arm_target_slew_violation_max_rad=slew_max))
    if len(checks)!=128:raise ValueError('Full TRAIN128 audit required')
    for path in [Path(__file__),a.checkpoint,history_path,traces/'manifest.json',ROOT/module]+[
            a.run/x for x in ('INPUTS.json','job.json','metrics.json','episodes.json','completed.json','execution_source_hashes.json')]:
        hashes[str(path.resolve())]=sha(path)
    result=dict(schema='native-learned-recovery-command-audit-v1',passed=True,run=str(a.run.resolve()),
        complete5=outcomes['complete5'],air=outcomes['air'],recovery=outcomes['recovery'],
        extra_mpc_or_vhsip_enabled=False,base_controller_and_persistent_recovery_retained=True,
        active_first_episode_ticks=sum(r['active_ticks'] for r in checks),
        cpu_vs_native_command_max_rad=max(r['cpu_vs_native_command_max_rad'] for r in checks),
        native_arm_target_slew_violation_max_rad=max(r['native_arm_target_slew_violation_max_rad'] for r in checks),
        command_tolerance_rad=5e-5,slew_tolerance_rad=5e-6,cases=checks,source_sha256=hashes,
        scope='Completed native TRAIN128 command mapping and causal activation; no claim of generalization or full independent RL')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','cases')}))


if __name__=='__main__':main()
