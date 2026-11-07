#!/usr/bin/env python3
"""Audit learner proposal -> MPC request -> final command in native collection.

Preserves the original frozen-base request separately from the learned student
proposal. A future replacement head must target final minus original base,
not mistake final minus student for its total correction coordinate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.control.learned_recovery_command_v1 import admit_native_recovery_head,NativeRecoveryCommandAdapter
from quadsmash.rl.episode_trace_v4 import unpack_public


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--mpc-command-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior audits')
    torch.set_num_threads(1)
    read=lambda name:json.loads((a.run/name).read_text())
    inputs=read('INPUTS.json');job=read('job.json');metrics=read('metrics.json')
    if job['status']!='SUCCEEDED' or not (a.run/'completed.json').exists():raise ValueError('Native collection incomplete')
    if (inputs.get('learned_recovery_mpc_correction') is not True or inputs['extra_recovery_teacher_enabled'] is not True or
        inputs['landing_mpc_cases']!=list(range(128)) or inputs['student_sampling_log_prob_available'] or
        inputs['correction_collection_is_ppo_buffer']):raise ValueError('Incorrect assisted collection declaration')
    if not metrics['learned_recovery_command']['extra_mpc_or_vhsip_teacher_enabled']:
        raise ValueError('Do not label assisted execution as teacher-off learning')
    audit=json.loads(a.mpc_command_audit.read_text())
    if not audit['passed'] or Path(audit['run']).resolve()!=a.run.resolve():raise ValueError('Matching MPC execution audit required')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed MPC evidence')
    if sha(a.checkpoint)!=inputs['learned_recovery_command_head_sha256']:raise ValueError('Wrong student checkpoint')
    source=read('execution_source_hashes.json')
    for name in ['quadsmash/control/learned_recovery_command_v1.py','quadsmash/control/teacher_corrected_recovery_command_v1.py']:
        if sha(ROOT/name)!=source[name]:raise ValueError('Executed adapter changed')
    limits=np.asarray(read('native_command_inputs.json')['joint_limits_rad']);width=limits[:,1]-limits[:,0]
    head=admit_native_recovery_head(torch.load(a.checkpoint,map_location='cpu',weights_only=False),
        inputs['policy_sha256'],inputs['protocol_sha256'],limits,.01)
    traces=a.run/'fixed_policy_traces';manifest=json.loads((traces/'manifest.json').read_text())
    with np.load(a.run/'learned_recovery_commands.npz',allow_pickle=False) as z:
        history={k:z[k].copy() for k in z.files}
    mpc=read('landing_mpc_records.json');requests={}
    for record in mpc:
        key=(int(record['case']),int(record['control_tick']))
        if key in requests:raise ValueError('Repeated teacher call in a native tick')
        requests[key]=record
    rows=[];hashes={};used=set()
    for entry in manifest['episodes']:
        case=entry['source_training_index'];lane=entry['env_id'];n=entry['samples']
        if case!=lane:raise ValueError('Original ordered TRAIN128 required')
        path=traces/entry['data_file']
        if sha(path)!=entry['data_sha256']:raise ValueError('Changed native trace')
        hashes[str(path.resolve())]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            public=unpack_public(z['packed_public']);actual=z['dispatched_actions'].copy()
        if not np.array_equal(actual,history['final_command'][:n,lane]):raise ValueError('Final record is not executed')
        if not history['first_episode'][:n,lane].all() or history['first_episode'][n:,lane].any():
            raise ValueError('First-episode mask differs')
        adapter=NativeRecoveryCommandAdapter(head,1,'cpu',limits)
        proposal_max=0.;request_max=0.;history_max=0.;slew_max=0.;delta_max=0.;changed=0;active_count=0
        for tick in range(n):
            previous=actual[tick-1:tick] if tick else np.zeros((1,18),np.float32)
            adapter.previous.copy_(torch.from_numpy(previous))
            proposed=adapter.action(torch.from_numpy(public[tick:tick+1]),
                torch.from_numpy(history['base_command'][tick,lane:lane+1])).numpy()[0]
            original=history['student_command'][tick,lane]
            proposal_max=max(proposal_max,float(np.max(abs(proposed-original)*width/2.)))
            history_max=max(history_max,float(np.max(abs(previous[0]-history['previous_command'][tick,lane])*width/2.)))
            delta_max=max(delta_max,float(np.max(abs(adapter.last_delta.numpy()[0]-history['learned_delta'][tick,lane]))))
            active=bool(adapter.active[0]);active_count+=active
            if active!=bool(history['active'][tick,lane]):raise ValueError('Causal eligibility differs')
            if not active and not np.array_equal(actual[tick],original):raise ValueError('Teacher acted before eligibility')
            if not np.array_equal(history['teacher_command_delta'][tick,lane],actual[tick]-original):
                raise ValueError('Paired delta does not describe executed correction')
            correction=float(np.max(abs(actual[tick]-original)*width/2.))
            flagged=bool(history['teacher_changed'][tick,lane]);changed+=flagged
            if abs(correction-1e-5)>2e-7 and flagged!=(correction>1e-5):raise ValueError('Wrong correction flag')
            if active:
                key=(case,tick)
                if key not in requests:raise ValueError('Eligible recovery step not recorded by teacher')
                used.add(key);record=requests[key]
                target=limits[:,0]+.5*(original+1.)*width
                request_max=max(request_max,float(np.max(abs(target-np.asarray(record['requested_student_joint_reference'])))))
                slew_max=max(slew_max,float(np.maximum(0.,abs(actual[tick,12:]-previous[0,12:])*width[12:]/2.-.1).max()))
        if proposal_max>5e-5 or request_max>5e-6 or history_max>5e-6 or slew_max>5e-6 or delta_max>2e-5:
            raise ValueError('Command-chain mismatch at case '+str(case))
        rows.append(dict(case=case,active_ticks=active_count,teacher_changed_ticks=changed,
            student_proposal_error_max_rad=proposal_max,mpc_request_error_max_rad=request_max,
            previous_final_command_error_max_rad=history_max,arm_slew_violation_max_rad=slew_max))
    if len(rows)!=128 or used!=set(requests):raise ValueError('Unaccounted cases or teacher calls')
    for f in [Path(__file__),a.checkpoint,a.mpc_command_audit,traces/'manifest.json']+[
            a.run/n for n in ('INPUTS.json','metrics.json','job.json','completed.json','learned_recovery_commands.npz','execution_source_hashes.json')]:
        hashes[str(f.resolve())]=sha(f)
    result=dict(schema='native-student-teacher-command-audit-v1',passed=True,run=str(a.run.resolve()),
        complete5=metrics['complete5'],extra_teacher_enabled=True,teacher_off_learning_result=False,
        student_log_prob_available=False,on_policy_ppo_buffer=False,positive_dataset_generated=False,
        active_ticks=sum(r['active_ticks'] for r in rows),teacher_changed_ticks=sum(r['teacher_changed_ticks'] for r in rows),
        maxima={k:max(r[k] for r in rows) for k in ('student_proposal_error_max_rad','mpc_request_error_max_rad',
            'previous_final_command_error_max_rad','arm_slew_violation_max_rad')},cases=rows,source_sha256=hashes,scope=__doc__)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_sha256','scope')}))


if __name__=='__main__':main()
