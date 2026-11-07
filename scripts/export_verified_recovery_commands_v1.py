#!/usr/bin/env python3
"""Export outcome-filtered native command sequences for recovery supervision.

Stores total position commands, not clipped inverses of the old residual. This
is supervised data from deterministic evaluation, not an on-policy PPO buffer.
No raw student residual or log_prob is fabricated. Failed outcomes stay in the
manifest for analysis and never enter positive command files.
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
from quadsmash.rl.recovery_command_dataset_v1 import recovery_eligibility,native_commands,previous_commands


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--command-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Do not overwrite an existing dataset')
    read=lambda n:json.loads((args.run/n).read_text())
    job=read('job.json');inputs=read('INPUTS.json');native=read('native_command_inputs.json')
    if job['status']!='SUCCEEDED' or job['exit_code']!=0:raise ValueError('Completed native run required')
    audit=json.loads(args.command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=args.run.resolve():
        raise ValueError('Require matching passed actual-command audit')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Audited input changed: '+path)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=json.loads(protocol_path.read_text());protocol['_root']=str(ROOT)
    summary,_=read_run(args.run,protocol)
    if summary['numerical_failures']:raise ValueError('Numerical failures in source execution')
    limits=np.asarray(native['joint_limits_rad']);rate=float(audit['arm_target_rate_rad_s'])
    metadata=read('fixed_policy_traces/manifest.json')
    if (metadata['action_interface']!='absolute-joint-limits-v1' or metadata['control_dt_s']!=.01 or
            metadata['provenance']['policy_sha256']!=inputs['policy_sha256']):
        raise ValueError('Trace interface/provenance mismatch')
    entries={e['source_training_index']:e for e in metadata['episodes']}
    if sorted(entries)!=list(range(128)):raise ValueError('Require original complete TRAIN128 trace manifest')
    rows=read('landing_mpc_records.json');by_case={}
    for row in rows:
        if row['episode_index']!=0:raise ValueError('First episode records required')
        key=(row['case'],row['control_tick'])
        if key in by_case:raise ValueError('Duplicate native correction record')
        by_case[key]=row
    stages={s['training_index']:s for s in summary['stages']}
    selected=sorted(k for k,s in stages.items() if s['recovery'])
    args.output.mkdir(parents=True)
    exported=[];max_command_error=0.;excluded=[]
    for case in range(128):
        if case not in selected:
            excluded.append(dict(case=case,reason='same-contact-flight recovery not successful',
                had_applied_correction=any(r['case']==case and r['applied_first_step'] for r in rows),
                outcome=stages[case]))
            continue
        entry=entries[case];source=args.run/'fixed_policy_traces'/entry['data_file']
        if sha(source)!=entry['data_sha256']:raise ValueError('Trace hash mismatch')
        with np.load(source,allow_pickle=False) as z:
            packed=z['packed_public'].copy();actions=z['dispatched_actions'].copy()
        public=unpack_public(packed)
        if sha_bytes(public)!=entry['full_public794_sha256']:raise ValueError('Lossless public history mismatch')
        eligible,elapsed=recovery_eligibility(public)
        normalized,target=native_commands(actions,limits)
        previous,previous_known=previous_commands(target)
        n=len(public)
        if len(actions)!=n:raise ValueError('Action/observation length mismatch')
        supervision=eligible & previous_known
        if not supervision.any():raise ValueError('Recovery outcome has no causal observed-hit/support window')
        slew=np.maximum(abs(target[1:,12:]-previous[1:,12:])-rate*.01,0.)
        if slew.max(initial=0.)>5e-6:raise ValueError('Actual arm command history violates source interface')
        requested=np.full((n,18),np.nan);teacher=np.full((n,18),np.nan)
        request_known=np.zeros(n,dtype=bool);teacher_known=np.zeros(n,dtype=bool)
        applied=np.zeros(n,dtype=bool);full_horizon=np.zeros(n,dtype=bool)
        # 0 excluded; 1 unmodified success reference; 2 successful executed
        # correction; 3 student fallback within a corrected successful episode.
        role=np.where(supervision,1,0).astype(np.uint8)
        for tick in range(n):
            r=by_case.get((case,tick))
            if r is None:continue
            if tick==0 or not previous_known[tick]:raise ValueError('Correction without known previous command')
            errors=[np.max(abs(target[tick]-np.asarray(r['final_dispatched_joint_target_rad']))),
                    np.max(abs(previous[tick]-np.asarray(r['previous_dispatched_target'])))]
            max_command_error=max(max_command_error,*map(float,errors))
            if max(errors)>5e-6:raise ValueError('Export command differs from native telemetry')
            requested[tick]=r['requested_student_joint_reference'];request_known[tick]=True
            applied[tick]=bool(r['applied_first_step']);full_horizon[tick]=bool(r['accepted'])
            if applied[tick]:
                teacher[tick]=r['admitted_joint_position_target_rad'];teacher_known[tick]=True
                if np.max(abs(teacher[tick]-target[tick]))>5e-6:
                    raise ValueError('Teacher admitted target was not the final command')
                if not eligible[tick]:raise ValueError('Applied recovery precedes the causal training gate')
            if supervision[tick]:role[tick]=2 if applied[tick] else 3
        outcome='complete5' if stages[case]['complete5'] else 'recovery_only'
        output=args.output/f'case_{case:03d}.npz'
        np.savez_compressed(output,packed_public=packed,native_action=normalized,
            native_joint_target_rad=target,previous_joint_target_rad=previous,
            previous_command_known=previous_known,recovery_eligible=eligible,
            elapsed_since_recovery_s=elapsed,supervision_mask=supervision,label_role=role,
            student_requested_joint_target_rad=requested,student_request_known=request_known,
            teacher_admitted_joint_target_rad=teacher,teacher_target_known=teacher_known,
            teacher_applied=applied,teacher_full_horizon_accepted=full_horizon)
        exported.append(dict(case=case,outcome=outcome,samples=n,supervision_rows=int(supervision.sum()),
            first_recovery_tick=int(np.flatnonzero(eligible)[0]),
            executed_correction_rows=int((supervision&applied).sum()),
            fallback_rows=int((role==3).sum()),unmodified_reference_rows=int((role==1).sum()),
            data_file=output.name,data_sha256=sha(output),source_trace=str(source.resolve()),
            source_trace_sha256=sha(source),complete5=stages[case]['complete5'],
            full_public794_sha256=entry['full_public794_sha256']))
    manifest=dict(schema='verified-recovery-native-command-dataset-v1',source=str(args.run.resolve()),
        selection='independently recomputed contact-flight recovery; complete5 and recovery-only labels separated',
        command_interface='total absolute joint targets and native normalized actions; not legacy residual labels',
        causal_gate='observed airborne hit, then phase774; latch until actual episode end',
        gate_uses_future_outcome=False,supervision_selection_uses_finished_outcome=True,
        raw_student_residual_available=False,student_log_prob_available=False,on_policy_ppo_buffer=False,
        teacher_off_learning_verified=False,formal_acceptance=False,generalization_test=False,
        control_dt_s=.01,arm_target_rate_rad_s=rate,joint_limits_rad=limits.tolist(),
        base_policy_sha256=inputs['policy_sha256'],protocol_sha256=inputs['protocol_sha256'],
        maximum_native_command_crosscheck_error_rad=max_command_error,
        positive_episodes=len(exported),complete5_episodes=sum(e['complete5'] for e in exported),
        recovery_only_episodes=sum(not e['complete5'] for e in exported),
        positive_rows=sum(e['supervision_rows'] for e in exported),
        executed_correction_rows=sum(e['executed_correction_rows'] for e in exported),
        corrected_complete5_cases=[e['case'] for e in exported if e['complete5'] and e['executed_correction_rows']],
        episodes=exported,excluded_outcomes=excluded,
        label_roles={'0':'excluded','1':'unmodified_success_reference','2':'executed_successful_correction',
                     '3':'student_fallback_inside_successful_corrected_recovery'},
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),args.command_audit,protocol_path,
            ROOT/'quadsmash/rl/recovery_command_dataset_v1.py',args.run/'episodes.json',
            args.run/'metrics.json',args.run/'INPUTS.json',args.run/'landing_mpc_records.json',
            args.run/'fixed_policy_traces/manifest.json']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k in ['positive_episodes','complete5_episodes',
        'recovery_only_episodes','positive_rows','executed_correction_rows','corrected_complete5_cases',
        'maximum_native_command_crosscheck_error_rad']}))


def sha_bytes(array):return hashlib.sha256(array.tobytes()).hexdigest()


if __name__=='__main__':main()
