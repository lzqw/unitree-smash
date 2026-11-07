#!/usr/bin/env python3
"""Check total-label reconstruction on finished native learner executions.

Teacher-off records are an explicit zero-additional-teacher control for this
coordinate audit, never relabelled as MPC intervention evidence.
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
from quadsmash.rl.paired_native_recovery_dataset_v1 import paired_labels


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--command-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier audit')
    audit=json.loads(a.command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=a.run.resolve():raise ValueError('Matching actual command audit required')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Audited evidence changed')
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text());protocol['_root']=str(ROOT)
    summary,_=read_run(a.run,protocol);stages={r['training_index']:r for r in summary['stages']}
    inputs=json.loads((a.run/'INPUTS.json').read_text());assisted=inputs['extra_recovery_teacher_enabled'] is True
    expected='native-student-teacher-command-audit-v1' if assisted else 'native-learned-recovery-command-audit-v1'
    if audit['schema']!=expected:raise ValueError('Wrong audit for execution mode')
    limits=np.asarray(json.loads((a.run/'native_command_inputs.json').read_text())['joint_limits_rad'])
    traces=a.run/'fixed_policy_traces';manifest=json.loads((traces/'manifest.json').read_text())
    with np.load(a.run/'learned_recovery_commands.npz',allow_pickle=False) as z:history={k:z[k].copy() for k in z.files}
    rows=[]
    for entry in manifest['episodes']:
        case=entry['source_training_index'];lane=entry['env_id'];n=entry['samples'];path=traces/entry['data_file']
        if case!=lane or sha(path)!=entry['data_sha256']:raise ValueError('Trace mapping or hash differs')
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public']);final=z['dispatched_actions'].copy()
        previous=np.vstack((np.zeros((1,18),np.float32),final[:-1]))
        student=history['student_command'][:n,lane] if assisted else final
        labels=paired_labels(public,history['base_command'][:n,lane],student,final,previous,
            history['active'][:n,lane],limits,recovery_success=stages[case]['recovery'],complete5=stages[case]['complete5'])
        rows.append(dict(case=case,complete5=stages[case]['complete5'],recovery=stages[case]['recovery'],
            positive_rows=int(labels['supervision_mask'].sum()),teacher_changed_rows=int(labels['teacher_changed'].sum()),
            reconstruction_error_max_rad=labels['reconstruction_error_max_rad']))
    if len(rows)!=128:raise ValueError('Full original TRAIN128 required')
    if any(r['positive_rows'] for r in rows if not r['recovery']):raise ValueError('Failure admitted as positive')
    sources=[Path(__file__),a.command_audit,a.run/'learned_recovery_commands.npz',traces/'manifest.json',
        ROOT/'quadsmash/rl/paired_native_recovery_dataset_v1.py']
    result=dict(schema='paired-recovery-label-coordinate-audit-v1',passed=True,extra_teacher_enabled=assisted,
        source_mode='assisted actual paired commands' if assisted else 'teacher-off zero-additional-teacher control',
        native_metrics={k:summary[k] for k in ('complete5','recovery','air')},
        positive_episodes=sum(r['positive_rows']>0 for r in rows),positive_rows=sum(r['positive_rows'] for r in rows),
        failed_positive_rows=0,teacher_changed_rows=sum(r['teacher_changed_rows'] for r in rows),
        reconstruction_error_max_rad=max(r['reconstruction_error_max_rad'] for r in rows),cases=rows,
        source_sha256={str(f.resolve()):sha(f) for f in sources},positive_dataset_generated=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_sha256')}))


if __name__=='__main__':main()
