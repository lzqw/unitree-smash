#!/usr/bin/env python3
"""Verify causal all128 intervention and unchanged original pre-intervention prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.observed_posthit_gate_v1 import ObservedPostHitGateV1
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','baseline','command-audit','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise ValueError('Preserve prior audit')
    read=lambda n:json.loads((a.run/n).read_text())
    inputs=read('INPUTS.json');job=read('job.json');metrics=read('metrics.json')
    if (job['status']!='SUCCEEDED' or job['exit_code']!=0 or
        inputs['landing_mpc_cases']!=list(range(128)) or not inputs['landing_mpc_causal_posthit_only'] or
        inputs['learned_recovery_command_head'] is not None or not inputs['extra_recovery_teacher_enabled']):
        raise ValueError('Completed original student all128 causal teacher run required')
    audit=json.loads(a.command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=a.run.resolve():raise ValueError('Matching command audit required')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Execution evidence changed')
    if inputs['policy_sha256']!=json.loads((a.baseline/'INPUTS.json').read_text())['policy_sha256']:
        raise ValueError('Original frozen policy differs')
    with np.load(a.run/'causal_teacher_commands.npz',allow_pickle=False) as z:history={k:z[k].copy() for k in z.files}
    records={(r['case'],r['control_tick']) for r in read('landing_mpc_records.json')}
    manifest=read('fixed_policy_traces/manifest.json');used=set();rows=[];sources=[]
    for entry in manifest['episodes']:
        case=entry['source_training_index'];lane=entry['env_id'];n=entry['samples']
        if lane!=case:raise ValueError('Original case/lane order required')
        path=a.run/'fixed_policy_traces'/entry['data_file'];sources.append(path)
        if sha(path)!=entry['data_sha256']:raise ValueError('Changed first-episode trace')
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public']);actual=z['dispatched_actions'].copy()
        if not np.array_equal(actual,history['final_command'][:n,lane]):raise ValueError('Final telemetry differs from actual commands')
        if not history['first_episode'][:n,lane].all() or history['first_episode'][n:,lane].any():
            raise ValueError('Episode mask mismatch')
        gate=ObservedPostHitGateV1(1,'cpu');eligible=[]
        for tick in range(n):eligible.append(bool(gate.update(torch.from_numpy(public[tick:tick+1]))[0]))
        active=np.asarray(eligible,dtype=bool)
        if not np.array_equal(active,history['active'][:n,lane]):raise ValueError('Noncausal or inconsistent gate')
        if not np.array_equal(actual[~active],history['base_command'][:n,lane][~active]):
            raise ValueError('Teacher changed inactive commands')
        used.update((case,int(t)) for t in np.flatnonzero(active))
        onset=int(np.flatnonzero(active)[0]) if active.any() else n
        basepath=a.baseline/'fixed_policy_traces'/entry['data_file'];sources.append(basepath)
        with np.load(basepath,allow_pickle=False) as z:
            bp=unpack_public(z['packed_public']);ba=z['dispatched_actions']
            if len(bp)<onset:raise ValueError('Baseline ended before intervention')
            obs_error=float(np.max(abs(public[:onset]-bp[:onset]),initial=0.))
            action_error=float(np.max(abs(actual[:onset]-ba[:onset]),initial=0.))
        if obs_error>1e-5 or action_error>1e-6:raise ValueError('Pre-intervention prefix changed: '+str(case))
        rows.append(dict(case=case,onset_tick=onset if active.any() else None,active_ticks=int(active.sum()),
                         prefix_observation_error_max=obs_error,prefix_action_error_max=action_error))
    if len(rows)!=128 or records!=used:raise ValueError('Missing or extra intervention calls')
    sources += [a.command_audit,Path(__file__),ROOT/'quadsmash/control/observed_posthit_gate_v1.py',
                a.run/'causal_teacher_commands.npz',a.run/'INPUTS.json',a.run/'metrics.json',a.run/'job.json']
    result=dict(passed=True,run=str(a.run.resolve()),baseline=str(a.baseline.resolve()),
        complete5=metrics['complete5'],teacher_off_learning_result=False,
        active_ticks=sum(r['active_ticks'] for r in rows),cases=rows,
        source_sha256={str(f.resolve()):sha(f) for f in sources})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_sha256']}))


if __name__=='__main__':main()
