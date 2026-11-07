#!/usr/bin/env python3
"""Audit causal dispatch separately from cross-run lane trajectory equality.

The strict v1 historical-prefix audit remains unchanged. This audit verifies
causal gating, inactive command passthrough, and the global trajectory before
any changed teacher command. Later per-lane divergence is explicitly reported,
not converted into historical equivalence or attributed to a unique cause.
"""
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
    changed=np.any(history['final_command']!=history['base_command'],axis=2) & history['first_episode']
    if np.any(changed & ~history['active']):raise ValueError('Inactive command altered')
    changed_ticks=np.flatnonzero(changed.any(axis=1))
    first_global_change=int(changed_ticks[0]) if len(changed_ticks) else len(changed)
    records={(r['case'],r['control_tick']) for r in read('landing_mpc_records.json')}
    baseline_metrics=json.loads((a.baseline/'metrics.json').read_text())
    ball_cases={r['training_index'] for r in baseline_metrics['complete5_case_outcomes'] if r['ball5_except_recovery']}
    success_cases={r['training_index'] for r in baseline_metrics['complete5_case_outcomes'] if r['complete5']}
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
            common=min(onset,len(bp))
            obs_delta=abs(public[:common]-bp[:common])
            action_delta=abs(actual[:common]-ba[:common])
            obs_error=float(np.max(obs_delta,initial=0.))
            action_error=float(np.max(action_delta,initial=0.))
            global_end=min(first_global_change+1,n,len(bp))
            if not np.array_equal(public[:global_end],bp[:global_end]):
                raise ValueError('Global pre-change observation differs: '+str(case))
            if not np.array_equal(actual[:min(first_global_change,n)],ba[:min(first_global_change,n)]):
                raise ValueError('Global pre-change command differs: '+str(case))
            differing=np.flatnonzero((obs_delta.max(axis=1)>1e-5) | (action_delta.max(axis=1)>1e-6))
            first_diff=int(differing[0]) if len(differing) else None
            previous_equal=(None if first_diff is None or first_diff==0 else
                bool(np.array_equal(public[first_diff-1],bp[first_diff-1]) and
                     np.array_equal(actual[first_diff-1],ba[first_diff-1])))
        prefix_equal=onset<=len(bp) and obs_error<=1e-5 and action_error<=1e-6
        rows.append(dict(case=case,onset_tick=onset if active.any() else None,active_ticks=int(active.sum()),
                         prefix_observation_error_max=obs_error,prefix_action_error_max=action_error,
                         historical_prefix_equal=prefix_equal,first_difference_tick=first_diff,
                         preceding_observation_and_command_equal=previous_equal,
                         baseline_complete5=case in success_cases,baseline_ball5=case in ball_cases))
    if len(rows)!=128 or records!=used:raise ValueError('Missing or extra intervention calls')
    sources += [a.command_audit,Path(__file__),ROOT/'quadsmash/control/observed_posthit_gate_v1.py',
                a.run/'causal_teacher_commands.npz',a.run/'INPUTS.json',a.run/'metrics.json',a.run/'job.json',a.baseline/'metrics.json']
    divergent=[r['case'] for r in rows if not r['historical_prefix_equal']]
    critical_equal=not any(not r['historical_prefix_equal'] and r['baseline_ball5'] for r in rows)
    result=dict(schema='causal-dispatch-integrity-v2',passed=True,
        causal_gate_and_command_mapping_verified=True,global_prechange_bitwise_equal=True,
        first_global_changed_command_tick=first_global_change,
        all_lane_historical_prefix_equal=not divergent,historical_prefix_divergent_cases=divergent,
        all_baseline_ball5_prefixes_equal=critical_equal,
        strict_v1_prefix_failure_not_overridden=True,
        indirect_physics_effect_not_excluded=True,unique_divergence_cause_established=False,
        run=str(a.run.resolve()),baseline=str(a.baseline.resolve()),
        complete5=metrics['complete5'],teacher_off_learning_result=False,
        active_ticks=sum(r['active_ticks'] for r in rows),cases=rows,
        source_sha256={str(f.resolve()):sha(f) for f in sources})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_sha256']}))


if __name__=='__main__':main()
