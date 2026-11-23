"""Sequential E1 v2 A8 blocks, frozen TRAIN64 checks and stable development selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from compare_supermimic_approach_ball_only import compare
from run_supermimic_approach_ball_only import (
    ROOT, evaluate_command, registered_options, train_command, verify_frozen_milestone, verify_training_milestone)


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(command, log, minutes):
    with log.open('ab') as stream:
        subprocess.run(
            ['timeout', '--signal=INT', '--kill-after=120s', f'{minutes}m', sys.executable, '-u', *command],
            cwd=ROOT, env=dict(os.environ, PYTHONPATH='.:scripts', QUADSMASH_ZSTD_ARCHIVE='1'),
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def quick_check(source, output, execution, seed):
    command = evaluate_command(source, output, True, execution, seed=seed, a0=False)
    command[command.index('--research-fixed128')] = '--approach-quick64'
    if not (output / 'completed.json').is_file():
        run(command, output.with_suffix('.log'), 40)
    metrics = json.loads((output / 'metrics.json').read_text())
    contract = metrics['task_contract']['approach_quick_panel']
    complete = json.loads((output / 'completed.json').read_text())
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (complete['episodes'] != 64 or metrics['evaluated_checkpoint_sha256'] != digest or metrics['numerical_failures']
            or any(metrics['optimizer_updates'].values()) or not metrics['actor_parameters_unchanged']):
        raise ValueError('Quick-check identity or native completion failed')
    summary = json.loads((output / 'episodes_summary.json').read_text())
    raw = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    if len(raw) != 64 or len(summary) != 64 or {r['case_id'] for r in raw} != {r['case_id'] for r in summary}:
        raise ValueError('Quick-check must have 64 paired original episodes')
    outcomes = {r['case_id']: bool(r['ball_side_complete_except_recovery']) for r in summary}
    return dict(panel_sha256=contract['sha256'], checkpoint=str(source), checkpoint_sha256=digest,
                success=sum(outcomes.values()), qualified=metrics['qualified_airborne_string_contact'],
                frame=sum(r['frame_hit'] and not r['string_hit'] for r in raw),
                low=sum(r['qualified_airborne_string_contact'] and 'speed_threshold' in r['rejection_reasons']
                        for r in summary),
                out_of_target=sum(r['crossed_net'] and not r['landed_target']
                                  and all(map(math.isfinite, r['first_landing_m'])) for r in raw),
                landed_target=sum(r['landed_target'] for r in raw), outcomes=outcomes)


def clearly_regressed(baseline, candidate):
    if baseline['panel_sha256'] != candidate['panel_sha256'] or baseline['outcomes'].keys() != candidate['outcomes'].keys():
        raise ValueError('Quick-check panels differ')
    # A small swing in successful cases alone is insufficient to stop.
    return (candidate['success'] <= baseline['success'] - 5 and (
        candidate['qualified'] <= baseline['qualified'] - 4 or candidate['low'] >= baseline['low'] + 4
        or candidate['frame'] >= baseline['frame'] + 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'source', 'history', 'output', 'restart-receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--stop-at', choices=(40, 48, 56, 64), type=int, default=40)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--development-baseline', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(args.restart_receipt.read_text())
    if receipt.get('ticks') != 64 or not isinstance(receipt.get('passed'), bool):
        raise ValueError('Inspect the native restart probe before training')
    registered = json.loads(args.registration.read_text())['command']
    execution = registered_options(registered, ('g07-execution-root',))[1]
    source = args.source.resolve(strict=True)
    source_metric = json.loads((args.development_baseline / 'metrics.json').read_text())
    with source.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != source_metric['evaluated_checkpoint_sha256']:
            raise ValueError('E1 v2 source differs from the evaluated A32 checkpoint')
    records = []
    regressions = 0
    best = None
    verified = {}
    state = args.output / 'queue.json'
    stage = 40
    retry = 0
    training_lanes, guard_lanes = 32, 16
    try:
        while stage <= args.stop_at:
            label = f'A{stage:03d}' + (f'_retry{retry}' if retry else '')
            folder = args.output / f'{label}_train'
            command = train_command(registered, source, folder, True, stop_at=stage, fresh=False,
                                    prioritize_flight=True, start_lanes=training_lanes, e1_v2=True, guard_lanes=guard_lanes,
                                    labeled_start_checkpoint=args.history.resolve(strict=True),
                                    labeled_snapshot_reserve=8)
            dump(state, dict(stage=stage, status='training', source=str(source), pid=os.getpid(),
                             native_collection='native_ground_prefix_replay', records=records))
            run(command, folder.with_suffix('.log'), 150)
            checkpoint, invocation = verify_training_milestone(folder, stage, source)
            verified[str(checkpoint)] = invocation
            throughput = folder / 'approach_v2_throughput.json'
            if throughput.is_file():
                trials = {row['lanes']: row for row in json.loads(throughput.read_text())['trials'] if 'lanes' in row}
                if (32 in trials and 64 in trials
                        and trials[64]['lane_ticks_per_second'] >= 1.1 * trials[32]['lane_ticks_per_second']
                        and trials[64]['api_peak_bytes'] < 10 * 1024**3):
                    # The sampler still requires 96 distinct labeled episodes
                    # and falls back to 32+16 if either batch cannot be filled.
                    training_lanes, guard_lanes = 64, 32
            dump(state, dict(stage=stage, status='quick_check', checkpoint=str(checkpoint), pid=os.getpid(),
                             records=records))
            if best is None:
                best = dict(stage=32, **quick_check(args.source, args.output / 'A032_quick64', execution, args.seed))
                records.append(best)
            result = dict(stage=stage, **quick_check(checkpoint, args.output / f'{label}_quick64', execution, args.seed))
            records.append(result)
            regressions = regressions + 1 if clearly_regressed(best, result) else 0
            if result['success'] > best['success']:
                best = result
            if regressions >= 2:
                if retry:
                    break
                checkpoint = Path(best['checkpoint'])
                stage = best['stage'] + 8
                source = checkpoint
                retry = 1
                regressions = 0
                continue
            source = checkpoint
            stage += 8
        evaluate = {r['checkpoint'] for r in records[-2:] if r['checkpoint'] in verified}
        if best['checkpoint'] in verified:
            evaluate.add(best['checkpoint'])
        development = {str(args.source): dict(valid_returns=source_metric['ball_side_complete_except_recovery'],
                                             checkpoint_sha256=source_metric['evaluated_checkpoint_sha256'])}
        for candidate in sorted(evaluate):
            candidate = Path(candidate)
            panel = args.output / f'{candidate.parent.name}_fixed128'
            run(evaluate_command(candidate, panel, True, execution, seed=args.seed, a0=False),
                panel.with_suffix('.log'), 40)
            score = verify_frozen_milestone(panel, candidate, verified[str(candidate)])
            score['paired_against_A32'] = compare(args.development_baseline, panel)
            development[str(candidate)] = score
        best_fixed = max(development, key=lambda checkpoint: (development[checkpoint]['valid_returns'],
                                                               checkpoint == str(args.source)))
        dump(args.output / 'development_selection.json', dict(
            panel='fixed128', scores=development, selected_checkpoint=best_fixed,
            independent512_scored=False))
        dump(state, dict(stage=records[-1]['stage'], status='finished', best_quick_checkpoint=best['checkpoint'],
                         checkpoint=str(source), development_selected_checkpoint=best_fixed, records=records))
    except BaseException as error:
        dump(state, dict(stage=stage, status='stopped', error=str(error), records=records))
        raise


if __name__ == '__main__':
    main()
