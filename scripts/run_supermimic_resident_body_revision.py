"""Continue a verified Body revision from A40 with a bounded native check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import perf_counter, sleep

from compare_supermimic_approach_ball_only import compare
from run_supermimic_approach_ball_only import (
    ROOT, evaluate_command, registered_options, train_command, verify_frozen_milestone,
    verify_training_milestone,
)
from run_supermimic_approach_v2 import block_timing, dump, paired_quick_result, quick_check, run


def gpu_memory():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'],
        text=True).strip().splitlines()
    if len(rows) != 1:
        raise ValueError('Resident memory check expects one GPU')
    used, free, total = map(int, rows[0].split(','))
    return dict(used_mib=used, free_mib=free, total_mib=total)


def stop_resident_for_memory(directory, ready, memory, required):
    evidence = directory / 'paired.json'
    dump(evidence, dict(scope='resident evaluator was not started', reason='insufficient_gpu_headroom',
                        memory=memory, required_free_mib=required))
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    dump(directory / 'decision.json', dict(update=ready['update'], checkpoint_sha256=ready['checkpoint_sha256'],
                                          paired_result_sha256=digest, action='stop'))


def evaluate_resident(directory, ready, baseline_path, baseline_panel, execution, seed, required):
    before = gpu_memory()
    if min(before['free_mib'], ready['gpu']['free_bytes'] // 2**20) < required:
        stop_resident_for_memory(directory, ready, before, required)
        return False
    command = [sys.executable, '-u', 'scripts/check_supermimic_resident_approach.py',
               '--boundary', str(directory), '--baseline-policy', str(baseline_path),
               '--baseline-panel', str(baseline_panel), '--execution-root', execution,
               '--seed', str(seed), '--required-free-mib', str(required)]
    started = perf_counter()
    peak = before['used_mib']
    samples = 0
    with (directory / 'evaluator.log').open('ab') as stream:
        with subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH='.:scripts'),
                              stdout=stream, stderr=subprocess.STDOUT) as evaluator:
            while evaluator.poll() is None:
                peak = max(peak, gpu_memory()['used_mib'])
                samples += 1
                sleep(2)
        dump(directory / 'gpu_telemetry.json', dict(parked=before, peak_used_mib=peak,
            samples=samples, seconds=perf_counter()-started, evaluator_exit=evaluator.returncode))
        if evaluator.returncode:
            raise subprocess.CalledProcessError(evaluator.returncode, command)
    return True


def resident_train(args, registered, execution, source, best_policy, best_panel, best, status, required):
    folder = args.output / 'train'
    command = train_command(
        registered, source, folder, True, stop_at=args.stop_at, fresh=False, prioritize_flight=True,
        start_lanes=32, guard_lanes=16, e1_v2=True, labeled_start_checkpoint=args.history,
        labeled_snapshot_reserve=8,
    )
    command.extend(('--approach-resident-eval-wait-seconds', '2700',
                    '--approach-resident-coverage-baseline', str(args.coverage_baseline)))
    if args.event_body_refresh:
        command.append('--approach-event-body-refresh')
    minutes = 90 if args.stop_at == 48 else 220
    command = ['timeout', '--signal=INT', '--kill-after=120s', f'{minutes}m', sys.executable, '-u', *command]
    records = [dict(stage=int(best_policy.stem[-3:]), **best)]
    processed = []
    stop_reason = None
    started = perf_counter()
    dump(status, dict(status='resident_training', source=str(source), records=records))
    with (args.output / 'train.log').open('ab') as stream:
        training = subprocess.Popen(command, cwd=ROOT, start_new_session=True, stdout=stream,
            stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH='.:scripts', QUADSMASH_ZSTD_ARCHIVE='1'))
        try:
            for stage in range(48, args.stop_at, 8):
                directory = folder / f'A{stage:03d}_resident_gate'
                ready_path = directory / 'ready.json'
                while not ready_path.is_file() and training.poll() is None:
                    sleep(2)
                if not ready_path.is_file():
                    break
                ready = json.loads(ready_path.read_text())
                dump(status, dict(status='resident_quick_check', stage=stage, source=str(source), records=records))
                if not evaluate_resident(directory, ready, best_policy, best_panel, execution, args.seed, required):
                    stop_reason = 'insufficient_gpu_headroom'
                    break
                result = json.loads((directory / 'paired.json').read_text())
                records.append(dict(stage=stage, **result['candidate'], paired=result['paired']))
                processed.append(directory)
                if result['paired']['regressed']:
                    stop_reason = 'native_regression_review'
                    break
                if result['candidate']['success'] > best['success']:
                    best, best_policy, best_panel = result['candidate'], directory / 'policy.pt', directory / 'quick64'
            code = training.wait()
        finally:
            if training.poll() is None:
                os.killpg(training.pid, signal.SIGINT)
                training.wait(timeout=120)
    if code:
        dump(status, dict(status='resident_training_stopped', code=code, records=records))
        raise subprocess.CalledProcessError(code, command)
    boundary = json.loads((folder / 'evaluation_boundary.json').read_text())
    stage = boundary['approach_accepted']
    checkpoint, invocation = verify_training_milestone(folder, stage, source)
    if invocation['actor_updates'] != stage - 40:
        raise ValueError('Resident continuation did not account for its accepted updates')
    timing = block_timing(folder, perf_counter() - started)
    timing['resident_eval_seconds'] = sum(json.loads((d / 'gpu_telemetry.json').read_text())['seconds'] for d in processed)
    dump(args.output / 'timing.json', timing)
    if stage < args.stop_at:
        dump(status, dict(status=stop_reason or 'resident_boundary_stopped', stage=stage, records=records,
                          checkpoint=str(checkpoint), best_quick_checkpoint=str(best_policy)))
        return
    panel = args.output / f'A{stage:03d}_quick64'
    result = quick_check(checkpoint, panel, execution, args.seed)
    pair = paired_quick_result(best, result, best_panel, panel)
    records.append(dict(stage=stage, **result, paired=pair))
    if not pair['regressed'] and result['success'] > best['success']:
        best_policy = checkpoint
    if args.stop_at == 48:
        dump(status, dict(status='native_regression_review' if pair['regressed'] else 'quick_check_complete',
                          stage=stage, records=records, checkpoint=str(checkpoint), best_quick_checkpoint=str(best_policy)))
        return
    dump(status, dict(status='fixed128', stage=stage, records=records, checkpoint=str(checkpoint)))
    fixed = args.output / 'A064_fixed128'
    run(evaluate_command(checkpoint, fixed, True, execution, seed=args.seed, a0=False), fixed.with_suffix('.log'), 40)
    score = verify_frozen_milestone(fixed, checkpoint, invocation)
    score['paired_against_A32'] = compare(args.development_baseline, fixed)
    dump(status, dict(status='fixed128_complete', stage=stage, records=records, checkpoint=str(checkpoint),
                      best_quick_checkpoint=str(best_policy), fixed128=score))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'pilot', 'history', 'quick-baseline', 'development-baseline', 'coverage-baseline', 'output'):
        parser.add_argument('--' + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--stop-at', type=int, choices=(48, 64), default=64)
    parser.add_argument('--event-body-refresh', action='store_true')
    parser.add_argument('--verified-a40-development', type=Path)
    args = parser.parse_args()
    if args.event_body_refresh and (args.stop_at != 48 or args.verified_a40_development is None):
        parser.error('The event-aligned refresh pilot requires a verified A40 development panel and stops at A48')
    registered = json.loads(args.registration.read_text())['command']
    execution = registered_options(registered, ('g07-execution-root',))[1]
    pilot = json.loads((args.pilot / 'status.json').read_text())
    if pilot['status'] not in ('quick_check_complete', 'fixed128_complete'):
        raise ValueError('A40 revision must finish its native checks before continuation')
    a32 = Path(pilot['native_before']['checkpoint'])
    source, invocation = verify_training_milestone(args.pilot / 'train', 40, a32)
    if invocation['actor_updates'] != 8 or source != Path(pilot['checkpoint']):
        raise ValueError('Expected a verified eight-update A40 revision')
    if args.verified_a40_development is not None:
        verify_frozen_milestone(args.verified_a40_development, source, invocation)
        compare(args.development_baseline, args.verified_a40_development)
    a32_score = quick_check(a32, args.quick_baseline, execution, args.seed)
    panel = args.pilot / 'A040_quick64'
    a40_score = quick_check(source, panel, execution, args.seed)
    if paired_quick_result(a32_score, a40_score, args.quick_baseline, panel)['regressed']:
        raise ValueError('A40 triggered the established native regression stop')
    samples = [json.loads(line) for line in (args.pilot / 'quick64_gpu_observation.log').read_text().splitlines()]
    if not samples:
        raise ValueError('No measured standalone TRAIN64 GPU footprint')
    required = max(sample['used_mib'] for sample in samples) + 1024
    args.output.mkdir()
    cache = args.output.parent / '.native_replay_cache'
    if cache.is_dir():
        (args.output / '.native_replay_cache').symlink_to(cache.resolve(), target_is_directory=True)
    dump(args.output / 'gpu_admission.json', dict(standalone_peak_used_mib=required-1024, reserved_headroom_mib=1024,
                                                   required_free_mib=required, samples=len(samples)))
    best = (source, panel, a40_score) if a40_score['success'] >= a32_score['success'] else (
        a32, args.quick_baseline, a32_score)
    resident_train(args, registered, execution, source, *best, args.output / 'status.json', required)


if __name__ == '__main__':
    main()
