"""Sequential E1 v2 A8 blocks, frozen TRAIN64 checks and stable development selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

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


def paired_quick_result(baseline, candidate, before_dir, after_dir):
    regressed = clearly_regressed(baseline, candidate)
    for name in ('initial_states.json', 'policy_execution.json', 'actuator_contract.json', 'physics.json', 'task_contract.json'):
        if json.loads((before_dir / name).read_text()) != json.loads((after_dir / name).read_text()):
            raise ValueError(f'TRAIN64 pair differs in {name}')

    def cases(path):
        raw = {row['case_id']: row for row in map(json.loads, (path / 'episodes.jsonl').read_text().splitlines())}
        score = json.loads((path / 'episodes_summary.json').read_text())
        result = {}
        for row in score:
            key = row['case_id']
            native = raw[key]
            string = bool(row['qualified_airborne_string_contact'])
            result[key] = dict(success=bool(row['ball_side_complete_except_recovery']), string=string,
                frame=bool(native['frame_hit'] and not native['string_hit']),
                low=string and 'speed_threshold' in row['rejection_reasons'],
                net_hit=bool(native['shuttle_net_contact']),
                out=bool(native['crossed_net'] and not native['landed_target']
                         and all(map(math.isfinite, native['first_landing_m']))))
        return result

    old, new = cases(before_dir), cases(after_dir)
    keys = sorted(old)
    return dict(regressed=regressed, transitions={
        name: [key for key in keys if old[key][left] and new[key][right]]
        for name, left, right in (('string_to_frame', 'string', 'frame'), ('frame_to_string', 'frame', 'string'))
    } | {
        name: dict(gained=[key for key in keys if not old[key][name] and new[key][name]],
                   lost=[key for key in keys if old[key][name] and not new[key][name]])
        for name in ('success', 'string', 'frame', 'low', 'net_hit', 'out')
    }, matched_speed={
        'adequate_to_low': [key for key in keys if old[key]['string'] and new[key]['string']
                            and not old[key]['low'] and new[key]['low']],
        'low_to_adequate': [key for key in keys if old[key]['string'] and new[key]['string']
                            and old[key]['low'] and not new[key]['low']]
    })


def block_timing(folder, elapsed):
    proposals = list(map(json.loads, (folder / 'training_history.jsonl').read_text().splitlines()))
    accepted = [row for row in proposals if row['update_accepted']]
    actor_seconds = sum(row['bptt_seconds'] for row in proposals)
    collection = list(folder.glob('*_native_collection_timing.json'))
    native = sum(json.loads(path.read_text())['through_archive_drain_seconds'] for path in collection)
    episode_log = folder / 'native_episodes.jsonl'
    episodes = list(map(json.loads, episode_log.read_text().splitlines())) if episode_log.is_file() else []
    contacts = {(row['env_id'], row['episode_index']) for row in episodes if row['qualified_airborne_string_contact']}
    return dict(wall_seconds=elapsed, actor_seconds=actor_seconds, native_collection_seconds=native,
        proposals=len(proposals), accepted=len(accepted), accepted_per_hour=len(accepted) * 3600 / elapsed,
        accepted_scales=[row['backtrack_alpha'] for row in accepted],
        distinct_native_qualified_contact_episodes=len(contacts),
        native_qualified_contact_episodes_per_hour=len(contacts) * 3600 / native if native else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registration', 'source', 'history', 'output', 'restart-receipt'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--stop-at', choices=(40, 48, 56, 64), type=int, default=40)
    parser.add_argument('--seed', type=int, default=882072)
    parser.add_argument('--development-baseline', type=Path, required=True)
    parser.add_argument('--resume-partial', type=Path, help='Exhausted A40 run directory containing its clean recovery snapshot')
    parser.add_argument('--retry-proposals', type=int, choices=(2, 4), default=4)
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
        partial_baseline = None
        previous_quick = None
        if args.resume_partial:
            original = args.resume_partial.resolve(strict=True)
            exhausted = json.loads((original / 'block_incomplete.json').read_text())
            prior = exhausted['accepted']
            if (exhausted['block'] != 'approach' or exhausted['target'] != 8
                    or (prior, exhausted['limit'], args.retry_proposals) not in ((6, 16, 4), (7, 20, 2))
                    or exhausted['attempts'] != exhausted['limit']):
                raise ValueError('Expected A38 with four new proposals, or A39 with two final proposals')
            partial_stage = 32 + prior
            a32_dir, partial_dir = args.output / 'A032_quick64', args.output / f'A{partial_stage:03d}_quick64'
            best = dict(stage=32, **quick_check(source, a32_dir, execution, args.seed))
            partial = original / f'approach_only_004_approach_candidate_{prior:04d}.pt'
            partial_baseline = dict(stage=partial_stage, **quick_check(partial, partial_dir, execution, args.seed))
            records.extend((best, partial_baseline))
            paired = paired_quick_result(best, partial_baseline, a32_dir, partial_dir)
            dump(args.output / f'A032_A{partial_stage:03d}_paired.json', paired)
            if paired['regressed']:
                dump(state, dict(stage=partial_stage, status='native_regression_review', records=records))
                return
            previous_quick = partial_baseline, partial_dir
            source = (original / 'interrupted.pt').resolve(strict=True)
        while stage <= args.stop_at:
            label = f'A{stage:03d}' + (f'_retry{retry}' if retry else '')
            recovering = args.resume_partial is not None and stage == 40 and retry == 0
            if recovering:
                label += f'_from{partial_stage:03d}'
            folder = args.output / f'{label}_train'
            command = train_command(registered, source, folder, True, stop_at=stage, fresh=False,
                                    prioritize_flight=True, start_lanes=training_lanes, e1_v2=True, guard_lanes=guard_lanes,
                                    labeled_start_checkpoint=args.history.resolve(strict=True),
                                    labeled_snapshot_reserve=8,
                                    retry_proposals=args.retry_proposals if recovering else 0)
            dump(state, dict(stage=stage, status='training', source=str(source), pid=os.getpid(),
                             native_collection='native_ground_prefix_replay', records=records))
            started = perf_counter()
            try:
                run(command, folder.with_suffix('.log'), 150)
            except subprocess.CalledProcessError:
                incomplete = folder / 'block_incomplete.json'
                if not incomplete.is_file():
                    raise
                stopped = json.loads(incomplete.read_text())
                if stopped['attempts'] != stopped['limit'] or stopped['block'] != 'approach':
                    raise
                candidates = list(folder.glob(f'*_approach_candidate_{stopped["accepted"]:04d}.pt'))
                if len(candidates) != 1:
                    raise
                dump(folder / 'v2_block_timing.json', block_timing(folder, perf_counter() - started))
                total = stage - stopped['target'] + stopped['accepted']
                result = dict(stage=total, **quick_check(candidates[0], args.output / f'{label}_partial_quick64', execution, args.seed))
                records.append(result)
                dump(state, dict(stage=total, status='proposal_budget_exhausted', checkpoint=str(candidates[0]),
                                 block=stopped, records=records))
                return
            dump(folder / 'v2_block_timing.json', block_timing(folder, perf_counter() - started))
            checkpoint, invocation = verify_training_milestone(folder, stage, source)
            verified[str(checkpoint)] = invocation
            dump(state, dict(stage=stage, status='quick_check', checkpoint=str(checkpoint), pid=os.getpid(),
                             records=records))
            if best is None:
                best = dict(stage=32, **quick_check(args.source, args.output / 'A032_quick64', execution, args.seed))
                records.append(best)
            quick_dir = args.output / f'{label}_quick64'
            result = dict(stage=stage, **quick_check(checkpoint, quick_dir, execution, args.seed))
            if previous_quick is not None:
                prior_result, prior_dir = previous_quick
                dump(args.output / f'A{prior_result["stage"]:03d}_A{stage:03d}_paired.json',
                     paired_quick_result(prior_result, result, prior_dir, quick_dir))
            previous_quick = result, quick_dir
            records.append(result)
            if recovering and (clearly_regressed(partial_baseline, result) or clearly_regressed(best, result)):
                dump(state, dict(stage=stage, status='native_regression_review', records=records))
                return
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
        if args.resume_partial:
            evaluate = {r['checkpoint'] for r in records if r['stage'] >= 48 and r['checkpoint'] in verified}
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
