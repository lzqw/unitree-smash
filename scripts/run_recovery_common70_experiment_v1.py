#!/usr/bin/env python3
"""Run recovery-head repair over the measured frozen approach candidate.

One persistent coordinator serializes training and frozen development tests.
It may wait for one explicitly identified live predecessor. It never restarts
a timed-out observation or overwrites an existing run. The original goal and
task definition are not marked complete by this experiment.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run,compare


def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def matches(pid,start_ticks):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[0]!='Z' and fields[19]==start_ticks
    except FileNotFoundError:
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--wait-start-ticks')
    args=parser.parse_args()
    if bool(args.wait_pid)!=bool(args.wait_start_ticks):
        raise ValueError('Identify both predecessor PID and process start ticks')
    if args.output.exists():
        raise ValueError('Use a fresh coordinator directory')
    args.output.mkdir(parents=True)
    lock=(args.output/'owner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reports=ROOT.parent/'unitree_smash/reports/G07_full_racket_v5'
    config=ROOT/'configs/g07_shared_target_ground64_attack_recovery_v24_frozen_approach_recovery_common70.json'
    best=reports/'frozen_v2_smooth_approach16_env5120_01/policy.pt'
    train=reports/'frozen_recovery_common70_16_env5120_01'
    precision=ROOT.parent/'unitree_smash/reports/G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
    state=dict(pid=os.getpid(),status='STARTING',goal_complete=False,configuration=str(config),
        training_output=str(train),source_policy=str(best))
    def save(**updates):
        state.update(updates,updated_at=datetime.now(timezone.utc).isoformat());atomic(args.output/'state.json',state)
    def guarded(stage,command,target,seconds,steps):
        if target.exists():
            raise ValueError('Existing run will not be restarted: '+str(target))
        cmd=[sys.executable,str(ROOT/'scripts/run_gpu_job.py'),'--goal','G07',
            '--budget-window','g07-full-racket-unlimited-20261024',
            '--max-seconds',str(seconds),'--env-steps-cap',str(steps),'--output',str(target),
            '--require-artifact','completed.json','--',sys.executable,'-u',*command]
        save(status=stage,command=cmd,current_output=str(target))
        with (args.output/(stage+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(child_guard_pid=child.pid)
            code=child.wait()
        save(child_guard_pid=None,last_exit_code=code)
        if code or not (target/'completed.json').exists():
            raise RuntimeError(stage+' failed; inspect '+str(target))
    try:
        if args.wait_pid:
            if not matches(args.wait_pid,args.wait_start_ticks):
                raise ValueError('Predecessor was not live when the coordinator started')
            save(status='WAITING_FOR_IDENTIFIED_PREDECESSOR',predecessor_pid=args.wait_pid,
                predecessor_start_ticks=args.wait_start_ticks)
            while matches(args.wait_pid,args.wait_start_ticks):
                time.sleep(5)
        train_args=[str(ROOT/'scripts/train_shared_target_ground64_success_retention_v1.py'),
            '--output',str(train),'--config',str(config),'--num-envs','5120','--updates','16',
            '--seed','882128','--initialize-actor',str(best),'--headless',
            '--base-checkpoint',str(reports/'shared_target_bias5_n128_01/best_candidate.pt'),
            '--body-source-evaluation',str(reports/'shared_target_bias5_best_cold128_01'),
            '--ground-source-evaluation',str(reports/'shared_target_bias5_ground64_cold128_03'),
            '--original-teacher-evaluation',str(reports/'shared_target_bias5_demonstrations_cold128_01'),
            '--precision-receipt',str(precision)]
        guarded('TRAINING',train_args,train,14400,5120*128*16)
        protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text());protocol['_root']=str(ROOT)
        baseline,_=read_run(reports/'frozen_v2_smooth_approach16_frozen128_seed882072_01',protocol)
        measured=[]
        for seed in [882072,882075]:
            target=reports/f'frozen_recovery_common70_16_frozen128_seed{seed}_01'
            evaluate=[str(ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_fasttelemetry_v1.py'),
                '--output',str(target),'--checkpoint',str(train/'policy.pt'),'--precision-receipt',str(precision),
                '--ground-xy-m','64','--record-first-episodes','--seed',str(seed),'--persistent-recovery-trial','--headless']
            guarded('EVALUATING_'+str(seed),evaluate,target,3600,102400)
            candidate,_=read_run(target,protocol)
            paired=dict(before={k:v for k,v in baseline.items() if k!='stages'},
                after={k:v for k,v in candidate.items() if k!='stages'},**compare(baseline,candidate))
            atomic(args.output/f'paired_seed{seed}.json',paired)
            measured.append(candidate)
            if candidate['complete5']<=baseline['complete5']:
                break
        save(status='EXPERIMENT_FINISHED',frozen_complete5=[m['complete5'] for m in measured],
            new_candidate_exceeds_baseline=all(m['complete5']>baseline['complete5'] for m in measured)
                and len(measured)==2,goal_complete=False)
    except BaseException as error:
        save(status='FAILED',error=repr(error),goal_complete=False)
        raise


if __name__=='__main__':
    main()
