#!/usr/bin/env python3
"""Select from completed native comparisons, then train and independently test.

Only the declared fresh-teacher recipe is automatic. If another measured
controller ranks better, publish that evidence for a new initialization choice.
All GPU children use the existing global guard; the predecessor is never stopped.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_landing_validation_sequence_v1 import atomic,sha,process_matches,verify_inputs,verify_completed
from scripts.export_complete_demonstrations_v1 import classify
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training


def score_run(directory,protocol):
    directory=Path(directory)
    complete=verify_completed(directory)
    steps=complete['env_steps']
    if type(steps) is not int or not 0<steps<=102400 or steps%128:
        raise ValueError('Require a valid original128 fixed-episode rollout')
    inputs=json.loads((directory/'INPUTS.json').read_text())
    cases,indices=training_cases(ROOT,protocol)
    if inputs['protocol_sha256']!=sha(ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json') or inputs['training_indices']!=indices:
        raise ValueError('Native comparison changed the protocol or distribution')
    rows=json.loads((directory/'episodes.json').read_text())
    summary=summarize_training(rows,cases,indices,protocol)
    reported=json.loads((directory/'metrics.json').read_text())
    if summary['numerical_failures'] or any(reported.get(k)!=v for k,v in summary.items()):
        raise ValueError('Native comparison is numerically invalid or disagrees with independent scoring')
    labels=[classify(row,protocol,5.) for row in rows]
    curriculum=[row['source_training_index'] for row,label in zip(rows,labels) if label['curriculum_complete']]
    rank=[summary['complete'],len(curriculum),summary['conditional_recovery_count'],summary['airborne_contacts']]
    return dict(output=str(directory),rank=rank,original_complete=summary['complete'],
        curriculum_complete5=len(curriculum),curriculum_complete5_cases=curriculum,
        recovery=summary['conditional_recovery_count'],air=summary['airborne_contacts'],
        numerical_failures=summary['numerical_failures'],
        evidence_sha256={name:sha(directory/name) for name in ['INPUTS.json','episodes.json','metrics.json','job.json','completed.json']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();plan=json.loads(args.plan.read_text());output=args.plan.parent
    verify_inputs(plan)
    protocol=json.loads(Path(plan['protocol']).read_text());protocol['_root']=str(ROOT)
    configuration=json.loads(Path(plan['training_config']).read_text())
    if (configuration['attack_speed_threshold_mps']!=5.
            or configuration['runner']['policy']['class_name']!='LandingExplorationActorCriticV1'
            or configuration['runner']['policy']['landing_exploration_scale']!=.15
            or configuration.get('support_assistance') is not None
            or configuration.get('landing_assistance') is not None
            or configuration['joint_scale_rad']!=[.4]*18):
        raise ValueError('Recipe must preserve the declared fresh all-joint curriculum')
    teacher=score_run(plan['teacher_output'],protocol)
    if teacher['rank']!=plan['expected_teacher_rank']:
        raise ValueError('Measured teacher reference changed')
    if args.preflight:
        print(json.dumps(dict(preflight=True,teacher=teacher,held_inputs=len(plan['held_inputs']),
            predecessor_live=process_matches(plan['predecessor']),gpu_started=False)),flush=True)
        return 0
    if (output/'state.json').exists():
        raise ValueError('Preserve previous followup lifecycle')
    if not process_matches(plan['predecessor']):
        raise ValueError('Register against the actually live predecessor process')
    lock=(output/'followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None;stopping=False
    state=dict(pid=os.getpid(),state='WAITING_FOR_NATIVE_COMPARISONS',
        plan_sha256=sha(args.plan),goal_complete=False,formal_acceptance=False)
    def save():
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(output/'state.json',state)
    def stop(_signal,_frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def run_guarded(name,command,target):
        nonlocal child
        if stopping:return False
        verify_inputs(plan)
        if Path(target).exists():raise ValueError('Never overwrite a previous native output: '+str(target))
        state.update(state=name,current_output=target);save()
        with (output/(name.lower()+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state['child_guard_pid']=child.pid;save();code=child.wait()
        child=None;state.pop('child_guard_pid',None)
        if stopping:state['state']='STOPPED';save();return False
        if code:raise ValueError(name+' failed with exit code '+str(code))
        verify_completed(target)
        return True
    try:
        save()
        while process_matches(plan['predecessor']):
            if stopping:state['state']='STOPPED_WHILE_WAITING';save();return 2
            save();time.sleep(5)
        previous=json.loads(Path(plan['predecessor_state']).read_text())
        if (previous['state']!='TRIALS_AND_COMPARISON_COMPLETE'
                or previous['plan_sha256']!=plan['predecessor_plan_sha256']
                or len(previous['completed_trials'])!=4):
            raise ValueError('Predecessor did not complete the declared four physical comparisons')
        verify_inputs(plan)
        extra=plan.get('pre_training_trial')
        if extra is not None:
            if not run_guarded('RUNNING_HEIGHT_ABLATION',extra['command'],extra['output']):return 2
        scores={name:score_run(path,protocol) for name,path in plan['comparisons'].items()}
        winner=max(scores,key=lambda name:tuple(scores[name]['rank']))
        atomic(output/'SELECTION.json',dict(scores=scores,winner=winner,
            ordering=['original8_complete','explicit5_curriculum_complete','original_recovery','original_air'],
            exact_tie_preference='Declared order, original teacher first',formal_acceptance=False))
        state.update(selection_file='SELECTION.json',selected_controller=winner)
        if winner!='teacher':
            state['state']='BETTER_CONTROLLER_REQUIRES_NEW_INITIALIZATION';save();return 0
        if not run_guarded('RUNNING_TRAINING',plan['training_command'],plan['training_output']):return 2
        training=Path(plan['training_output']);completed=verify_completed(training)
        metrics=json.loads((training/'metrics.json').read_text())
        if (completed['env_steps']!=plan['expected_training_steps']
                or metrics['completed_updates']!=plan['expected_updates']
                or metrics['history'][-1]['update']!=plan['expected_updates']
                or sha(training/'policy.pt')!=metrics['policy_sha256']):
            raise ValueError('Training completion differs from declared fresh run')
        if metrics['history'][-1]['numerical_failures']:
            raise ValueError('Training numerical failures require diagnosis')
        state['training_policy_sha256']=metrics['policy_sha256'];save()
        if not run_guarded('RUNNING_FIXED_EVALUATION',plan['evaluation_command'],plan['evaluation_output']):return 2
        checked=score_run(plan['evaluation_output'],protocol)
        atomic(output/'INDEPENDENT_RESULT.json',dict(before=scores['teacher'],after=checked,
            goal_complete=False,formal_acceptance=False,
            scope='Fixed TRAIN128 development result; original8 and explicit5 scores kept separate'))
        state.update(state='TRAINING_AND_INDEPENDENT_EVALUATION_COMPLETE',result_file='INDEPENDENT_RESULT.json');save()
        return 0
    except BaseException:
        state.update(state='FOLLOWUP_FAILED',error=traceback.format_exc());save();raise
    finally:
        lock.close()


if __name__=='__main__':
    raise SystemExit(main())
