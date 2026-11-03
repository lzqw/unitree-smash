#!/usr/bin/env python3
"""Audit paired finished128 evaluations under the unchanged 5m/s definition.

Keep the historical complete8 score separate. Report exact case transitions,
value of new contacts, and the measured state at the first post-hit support
handoff. This neither reads formal panels nor rewrites their scoring contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.export_complete_demonstrations_v1 import classify
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete5_summary(rows,protocol):
    labels=[classify(row,protocol,5.) for row in rows]
    stages=[]
    for row,label in zip(rows,labels):
        ball=bool(label['outgoing_speed_mps'] is not None and label['outgoing_speed_mps']>=5.
            and all(v for k,v in label['criteria_except_speed'].items() if k!='same_flight_stable_recovery'))
        stages.append(dict(training_index=row['source_training_index'],
            air=bool(label['original_outcomes']['airborne_contact']),
            recovery=bool(label['original_outcomes']['conditional_recovery']),
            complete5=bool(label['curriculum_complete']),ball5_except_recovery=ball,
            complete8=bool(label['original_complete']),rejection_reasons=label['rejection_reasons']))
    return dict(episodes=len(rows),complete5=sum(x['complete5'] for x in stages),
        complete5_rate=sum(x['complete5'] for x in stages)/len(rows),
        complete8=sum(x['complete8'] for x in stages),air=sum(x['air'] for x in stages),
        recovery=sum(x['recovery'] for x in stages),
        ball5_except_recovery=sum(x['ball5_except_recovery'] for x in stages),
        numerical_failures=sum(r['numerical_failures'] for r in rows),
        primary_metric='complete5',speed_threshold_mps=5.,stages=stages)


def read_run(source,protocol):
    source=Path(source)
    inputs=json.loads((source/'INPUTS.json').read_text())
    completed=json.loads((source/'completed.json').read_text())
    metrics=json.loads((source/'metrics.json').read_text())
    rows=json.loads((source/'episodes.json').read_text())
    cases,indices=training_cases(ROOT,protocol)
    summary=summarize_training(rows,cases,indices,protocol)
    if ((source/'error.txt').exists() or completed['env_steps']!=102400
            or inputs['training_indices']!=indices or any(inputs['reset_noise_amplitudes'].values())
            or inputs['protocol_sha256']!=sha(ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json')
            or metrics['actual_control_env_steps']!=102400
            or any(metrics.get(k)!=v for k,v in summary.items())):
        raise ValueError('Require finished, intact original128 zero-noise evaluation')
    result=complete5_summary(rows,protocol)
    result.update(output=str(source.resolve()),policy_sha256=inputs['policy_sha256'],
        evaluation_seed=inputs['reset_noise_seed'],target_records=sum(r['landed_target'] for r in rows),
        source_sha256={n:sha(source/n) for n in ['INPUTS.json','episodes.json','metrics.json','completed.json']})
    initial=json.loads((source/'initial_states.json').read_text())
    return result,initial


def handoff_states(source):
    root=Path(source)/'fixed_policy_traces'
    manifest=json.loads((root/'manifest.json').read_text())
    result={}
    for entry in manifest['episodes']:
        path=root/entry['data_file']
        if sha(path)!=entry['data_sha256']:
            raise ValueError('Handoff trace changed')
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public'])
        observed_hit=np.flatnonzero(public[:,773]>.5)
        if not len(observed_hit):
            continue
        supported=np.flatnonzero((public[:,774]>.5)&(np.arange(len(public))>observed_hit[0]))
        if not len(supported):
            continue
        tick=int(supported[0]); frame=public[max(0,tick-1)]
        result[entry['source_training_index']]=dict(
            first_posthit_phase774_s=float(public[tick,775]),pre_support_sample_s=float(frame[775]),
            root_velocity_b_mps=frame[:3].tolist(),root_angular_velocity_b_rad_s=frame[3:6].tolist(),
            gravity_b=frame[6:9].tolist(),foot_contact=frame[57:61].tolist(),
            joint_position_offset_rad=frame[9:27].tolist(),joint_velocity_rad_s=frame[27:45].tolist())
    return result


def compare(before,after):
    fields=['complete5','air','recovery','ball5_except_recovery']
    transitions={}
    for field in fields:
        a={x['training_index'] for x in before['stages'] if x[field]}
        b={x['training_index'] for x in after['stages'] if x[field]}
        transitions[field]=dict(before=len(a),after=len(b),retained=sorted(a&b),lost=sorted(a-b),gained=sorted(b-a))
    noncontact={x['training_index'] for x in before['stages'] if not x['air']}
    value={field:sorted(x['training_index'] for x in after['stages'] if x['training_index'] in noncontact and x[field])
           for field in ['air','ball5_except_recovery','recovery','complete5']}
    return dict(stage_transitions=transitions,old_noncontact_cohort=dict(total=len(noncontact),**value))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior comparisons')
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text());protocol['_root']=str(ROOT)
    before,bi=read_run(args.before,protocol);after,ai=read_run(args.after,protocol)
    if len(bi)!=128 or len(ai)!=128:
        raise ValueError('Missing initial states')
    errors={field:max(float(np.abs(np.asarray(a[field])-np.asarray(b[field])).max()) for a,b in zip(bi,ai))
            for field in ['root_state_local','joint_position_rad','joint_velocity_rad_s']}
    if any(v>1e-5 for v in errors.values()):
        raise ValueError('Paired initial conditions differ')
    result=dict(schema='paired-frozen-complete5-audit-v2',before=before,after=after,
        paired_initial_max_error=errors,**compare(before,after),
        before_handoff_states=handoff_states(args.before),after_handoff_states=handoff_states(args.after),
        formal_acceptance=False,generalization_test=False,
        scope='Paired fixed128 TRAIN development evidence; original8 fields retained separately.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
    print(json.dumps(dict(before=before['complete5'],after=after['complete5'],
        **{k:result[k] for k in ['stage_transitions','old_noncontact_cohort']})))


if __name__=='__main__':
    main()
