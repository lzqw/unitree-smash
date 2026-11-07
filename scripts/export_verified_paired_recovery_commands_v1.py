#!/usr/bin/env python3
"""Export actual successful student/MPC commands in replacement-head coordinates.

Only completed assisted native runs with a matching full command-chain audit
are eligible. Failed episodes remain negative outcome metadata, not experts.
No inferred log probabilities, fabricated teacher commands or automatic fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run
from quadsmash.rl.episode_trace_v4 import unpack_public
from quadsmash.rl.paired_native_recovery_dataset_v1 import SCHEMA,FEATURES,paired_labels,load_paired_episodes


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--paired-command-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier datasets')
    read=lambda name:json.loads((a.run/name).read_text())
    audit=json.loads(a.paired_command_audit.read_text());inputs=read('INPUTS.json');job=read('job.json')
    if (audit.get('schema')!='native-student-teacher-command-audit-v1' or not audit['passed'] or
        Path(audit['run']).resolve()!=a.run.resolve() or not audit['extra_teacher_enabled'] or
        inputs.get('learned_recovery_mpc_correction') is not True or
        job['status']!='SUCCEEDED' or job['exit_code']!=0):
        raise ValueError('Completed actually assisted execution and full paired audit required')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Audited evidence changed: '+path)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(ROOT)
    summary,_=read_run(a.run,protocol)
    if summary['numerical_failures']:raise ValueError('Numerical failures in source execution')
    stages={r['training_index']:r for r in summary['stages']}
    traces=a.run/'fixed_policy_traces';trace_manifest=json.loads((traces/'manifest.json').read_text())
    if trace_manifest['action_interface']!='absolute-joint-limits-v1' or trace_manifest['control_dt_s']!=.01:
        raise ValueError('Wrong native interface')
    limits=np.asarray(read('native_command_inputs.json')['joint_limits_rad'])
    with np.load(a.run/'learned_recovery_commands.npz',allow_pickle=False) as z:history={k:z[k].copy() for k in z.files}
    records={(int(r['case']),int(r['control_tick'])):r for r in read('landing_mpc_records.json')}
    entries={r['source_training_index']:r for r in trace_manifest['episodes']}
    if sorted(entries)!=list(range(128)):raise ValueError('Require original full TRAIN128')
    a.output.mkdir(parents=True);exported=[];excluded=[];max_error=0.
    for case in range(128):
        entry=entries[case];lane=entry['env_id'];n=entry['samples']
        if case!=lane:raise ValueError('Case/lane mapping differs')
        if not stages[case]['recovery']:
            excluded.append(dict(case=case,outcome=stages[case],positive_rows=0,
                teacher_changed_ticks=int(history['teacher_changed'][:n,lane].sum())))
            continue
        source=traces/entry['data_file']
        if sha(source)!=entry['data_sha256']:raise ValueError('Trace changed')
        with np.load(source,allow_pickle=False) as z:
            packed=z['packed_public'].copy();public=unpack_public(packed);final=z['dispatched_actions'].copy()
        if hashlib.sha256(public.tobytes()).hexdigest()!=entry['full_public794_sha256']:
            raise ValueError('Observation sequence changed')
        base=history['base_command'][:n,lane];student=history['student_command'][:n,lane]
        previous=history['previous_command'][:n,lane];active=history['active'][:n,lane]
        labels=paired_labels(public,base,student,final,previous,active,limits,
            recovery_success=True,complete5=stages[case]['complete5'])
        mask=labels['supervision_mask']
        if not mask.any():raise ValueError('Successful recovery has no causal supervision window')
        applied=np.zeros(n,dtype=bool);full_horizon=np.zeros(n,dtype=bool)
        for tick in np.flatnonzero(active):
            r=records[(case,int(tick))];applied[tick]=bool(r['applied_first_step']);full_horizon[tick]=bool(r['accepted'])
        max_error=max(max_error,labels.pop('reconstruction_error_max_rad'))
        path=a.output/f'case_{case:03d}.npz'
        np.savez_compressed(path,packed_public=packed,original_base_command=base,student_command=student,
            final_command=final,previous_final_command=previous,teacher_applied=applied,
            teacher_full_horizon_accepted=full_horizon,**labels)
        exported.append(dict(case=case,complete5=stages[case]['complete5'],
            outcome='complete5' if stages[case]['complete5'] else 'recovery_only',samples=n,
            supervision_rows=int(mask.sum()),teacher_changed_rows=int((mask&labels['teacher_changed']).sum()),
            teacher_applied_rows=int((mask&applied).sum()),data_file=path.name,data_sha256=sha(path),
            full_public794_sha256=entry['full_public794_sha256'],source_trace=str(source.resolve()),source_trace_sha256=sha(source)))
    files=[Path(__file__),a.paired_command_audit,protocol_path,ROOT/'quadsmash/rl/paired_native_recovery_dataset_v1.py',
        a.run/'learned_recovery_commands.npz',a.run/'landing_mpc_records.json',a.run/'episodes.json',traces/'manifest.json']
    manifest=dict(schema=SCHEMA,feature_semantics=FEATURES,source_run=str(a.run.resolve()),
        label_coordinate='(actual_final_native_command-original_frozen_base_command)/2',
        teacher_increment_is_total_label=False,student_log_prob_available=False,on_policy_ppo_buffer=False,
        positive_selection='independent original same-flight recovery; complete5 and recovery-only separated',
        failed_episodes_admitted=0,gate_uses_future_outcome=False,selection_uses_finished_outcome=True,
        control_dt_s=.01,arm_target_rate_rad_s=10.,joint_limits_rad=limits.tolist(),
        base_policy_sha256=inputs['policy_sha256'],protocol_sha256=inputs['protocol_sha256'],
        source_student_head_sha256=inputs['learned_recovery_command_head_sha256'],
        positive_episodes=len(exported),empty_dataset_not_trainable=not bool(exported),
        complete5_episodes=sum(e['complete5'] for e in exported),
        recovery_only_episodes=sum(not e['complete5'] for e in exported),
        positive_rows=sum(e['supervision_rows'] for e in exported),
        teacher_changed_rows=sum(e['teacher_changed_rows'] for e in exported),
        maximum_label_reconstruction_error_rad=max_error,episodes=exported,excluded_outcomes=excluded,
        student_training_performed=False,teacher_off_learning_verified=False,formal_acceptance=False,
        source_sha256={str(f.resolve()):sha(f) for f in files})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if exported:
        _,loaded,_=load_paired_episodes(a.output)
        if sum(len(e['target']) for e in loaded)!=manifest['positive_rows']:raise ValueError('Loader row count differs')
    report=dict(passed=True,positive_episodes=len(exported),positive_rows=manifest['positive_rows'],
        trainable=bool(exported),manifest_sha256=sha(a.output/'manifest.json'),student_training_performed=False)
    (a.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
