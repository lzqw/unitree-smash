#!/usr/bin/env python3
"""Summarize a finished teacher-off learned head and its executed corrections."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--command-audit',type=Path,required=True)
    p.add_argument('--student-comparison',type=Path,required=True)
    p.add_argument('--teacher-comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier evidence')
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text());protocol['_root']=str(ROOT)
    outcome,_=read_run(a.run,protocol)
    audit=json.loads(a.command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=a.run.resolve():raise ValueError('Matching native audit required')
    for name,expected in audit['source_sha256'].items():
        if sha(name)!=expected:raise ValueError('Audited source changed: '+name)
    comparisons={}
    for label,path in [('student',a.student_comparison),('teacher',a.teacher_comparison)]:
        j=json.loads(path.read_text())
        if Path(j['after']['output']).resolve()!=a.run.resolve():raise ValueError('Comparison is for another candidate')
        for side in ('before','after'):
            for name,expected in j[side]['source_sha256'].items():
                if sha(Path(j[side]['output'])/name)!=expected:raise ValueError('Comparison source changed')
        comparisons[label]=j['stage_transitions']
    native=json.loads((a.run/'native_command_inputs.json').read_text());limits=np.asarray(native['joint_limits_rad'])
    episodes={r['source_training_index']:r for r in json.loads((a.run/'episodes.json').read_text())}
    with np.load(a.run/'learned_recovery_commands.npz') as z:
        active=z['active']&z['first_episode'];delta=(z['final_command']-z['base_command'])*(limits[:,1]-limits[:,0])/2.
    rows=[]
    for case in range(128):
        ticks=np.flatnonzero(active[:,case]);d=abs(delta[ticks,case]);physical=episodes[case]['contact_flight_recovery']
        rows.append(dict(case=case,active_ticks=len(ticks),onset_tick=int(ticks[0]) if len(ticks) else None,
            onset_command_delta_max_rad=float(d[0].max()) if len(ticks) else None,
            first_second_command_delta_max_rad=float(d[:100].max()) if len(ticks) else None,
            active_command_delta_p95_rad=float(np.quantile(d,.95)) if len(ticks) else None,
            body_landing=physical['body_landing'],stable_recovery_s=physical['recovery_s'],
            terminal_root_height_m=physical['terminal_root_height_m']))
    lost=set(comparisons['student']['complete5']['lost'])
    paths=[Path(__file__),a.command_audit,a.student_comparison,a.teacher_comparison]+[
        a.run/x for x in ('learned_recovery_commands.npz','native_command_inputs.json','episodes.json','metrics.json','job.json')]
    result=dict(schema='native-recovery-learning-summary-v1',run=str(a.run.resolve()),
        metrics={k:outcome[k] for k in ('complete5','air','recovery','ball5_except_recovery','numerical_failures')},
        transitions=comparisons,lost_student_successes_with_body_landing=sum(r['case'] in lost and r['body_landing'] for r in rows),
        command_audit={k:v for k,v in audit.items() if k not in ('source_sha256','cases','records')},
        cases=rows,source_sha256={str(f.resolve()):sha(f) for f in paths},
        scope='Actual command deltas relative to frozen base evaluated on candidate states; not a causal decomposition, counterfactual teacher label, pure RL or generalization result.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['metrics']))


if __name__=='__main__':main()
