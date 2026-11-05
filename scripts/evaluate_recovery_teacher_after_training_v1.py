#!/usr/bin/env python3
"""Wait for one identified training guard, then evaluate its student twice.

No training is restarted, no in-progress checkpoint is evaluated, and no
teacher action is enabled by this coordinator. All native evaluation uses
the existing GPU0 authorization/accounting wrapper.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_frozen_complete5_v2 import read_run, compare


def matches(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == start_ticks
    except FileNotFoundError:
        return False


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--wait-start-ticks', required=True)
    args = parser.parse_args()
    if args.output.exists() or not matches(args.wait_pid, args.wait_start_ticks):
        raise ValueError('Require a fresh coordinator directory and a verified live predecessor')
    args.output.mkdir(parents=True)
    owner = (args.output/'owner.lock').open('x')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = args.training_output.resolve()
    reports = source.parent
    prefix = source.name.removesuffix('_env5120_01')
    targets = {seed:reports/f'{prefix}_frozen128_seed{seed}_01' for seed in [882072, 882075]}
    if any(path.exists() for path in targets.values()):
        raise ValueError('Never overwrite or automatically restart an evaluation')
    precision = reports.parent/'G07_full_racket_v4/origin_precision_after_aim_queue01/PRECISION_ADMISSION.json'
    state = dict(pid=os.getpid(), training_output=str(source),
        predecessor_pid=args.wait_pid, predecessor_start_ticks=args.wait_start_ticks,
        status='WAITING_FOR_IDENTIFIED_TRAINING_GUARD', goal_complete=False,
        teacher_disabled_for_evaluation=True, evaluated_results=[])
    def save(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        atomic(args.output/'state.json', state)
    save()
    try:
        while matches(args.wait_pid, args.wait_start_ticks):
            time.sleep(5)
        job = json.loads((source/'job.json').read_text())
        completion = json.loads((source/'completed.json').read_text())
        if (job['status'] != 'SUCCEEDED' or job['exit_code'] != 0
                or not (source/'policy.pt').is_file() or (source/'error.txt').exists()
                or completion.get('env_steps') != 5120 * 128 * 16):
            raise ValueError('Training did not successfully finish the declared16 updates')
        protocol = json.loads((ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text())
        protocol['_root'] = str(ROOT)
        evaluated = []
        for seed, target in targets.items():
            if target.exists():
                raise ValueError('Evaluation target appeared; inspect rather than restart: '+str(target))
            command = [sys.executable, str(ROOT/'scripts/run_gpu_job.py'), '--goal', 'G07',
                '--budget-window', 'g07-full-racket-unlimited-20261024', '--max-seconds', '3600',
                '--env-steps-cap', '102400', '--output', str(target), '--require-artifact', 'completed.json',
                '--', sys.executable, '-u',
                str(ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_fasttelemetry_v1.py'),
                '--output', str(target), '--checkpoint', str(source/'policy.pt'),
                '--precision-receipt', str(precision), '--ground-xy-m', '64', '--record-first-episodes',
                '--seed', str(seed), '--persistent-recovery-trial', '--headless']
            save(status='EVALUATING_'+str(seed), command=command, current_output=str(target))
            with (args.output/f'evaluate_{seed}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                save(evaluation_guard_pid=child.pid)
                code = child.wait()
            save(evaluation_guard_pid=None, last_exit_code=code)
            if code or not (target/'completed.json').is_file():
                raise RuntimeError('Frozen evaluation failed; inspect '+str(target))
            before, _ = read_run(reports/f'frozen_v2_smooth_approach16_frozen128_seed{seed}_01', protocol)
            after, _ = read_run(target, protocol)
            paired = dict(before=before, after=after, **compare(before, after),
                scope='Fixed128 TRAIN regression, not generalization or final acceptance')
            atomic(args.output/f'paired_seed{seed}.json', paired)
            evaluated.append(dict(seed=seed, complete5=after['complete5'], baseline_complete5=before['complete5'],
                air=after['air'], recovery=after['recovery'], numerical_failures=after['numerical_failures'],
                output=str(target), transitions=paired['stage_transitions']['complete5']))
            save(evaluated_results=evaluated)
        save(status='EXPERIMENT_EVALUATED',
            exceeds_baseline_both_seeds=all(r['complete5'] > r['baseline_complete5']
                and r['numerical_failures'] == 0 for r in evaluated),
            goal_complete=False, baseline_automatically_replaced=False)
    except BaseException as error:
        save(status='FAILED', error=repr(error), goal_complete=False)
        raise


if __name__ == '__main__':
    main()
