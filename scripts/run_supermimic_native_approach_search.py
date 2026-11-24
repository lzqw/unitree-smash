"""Run eight CEM updates from a measured baseline, then score the retained policy once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from quadsmash.rl.supermimic_native_approach_search import ARM_BIAS_LIMIT, actor_with_bias, rank_key
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_run_guard_v1 import atomic_torch_save
from run_supermimic_approach_ball_only import evaluate_command, registered_options, train_command
from run_supermimic_approach_v2 import ROOT, dump, paired_quick_result, quick_check, run

GENERATIONS = 8
NATIVE_SEED = 882072
SEARCH_SEED = 882091


def read(path):
    return json.loads(path.read_text())


def fresh_path(path):
    index = 0
    while path.exists():
        index += 1
        path = path.with_name(f'{path.name.split("_retry", 1)[0]}_retry{index:02d}')
    return path


def file_sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wait_json(path, process, seconds=900):
    deadline = time.monotonic() + seconds
    while not path.is_file():
        if process.poll() is not None:
            raise RuntimeError(f'Frozen WM screener exited with status {process.returncode} before {path.name}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Frozen WM screener did not produce {path.name}')
        time.sleep(.5)
    return read(path)


def proposals(incumbent, center, scale, generator):
    rows = [tuple(incumbent)]
    for direction in generator.normal(size=(3, 6)):
        for sign in (1., -1.):
            row = tuple(np.clip(center + sign * scale * direction, -ARM_BIAS_LIMIT, ARM_BIAS_LIMIT).tolist())
            if row not in rows:
                rows.append(row)
    if len(rows) < 2:
        raise ValueError('CEM generated no distinct challengers')
    return [list(row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'prepared', 'registration', 'history-checkpoint', 'risk-history',
                 'directed-collection', 'baseline-panel', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Resume from the last committed generation')
    args = parser.parse_args()
    output = args.output.resolve()
    source = args.source.resolve(strict=True)
    prepared = args.prepared.resolve(strict=True)
    registered = read(args.registration)['command']
    execution = registered_options(registered, ('g07-execution-root',))[1]
    preparation = read(prepared.parent/'approach_prepared.json')
    original = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    source_identity = dict(checkpoint_sha256=file_sha(source),
                           actor_sha256=tensor_digest(original['actor'].items()),
                           prepared_sha256=file_sha(prepared))
    if (preparation['actor_sha256'] != source_identity['actor_sha256']
            or preparation['checkpoint_sha256'] != source_identity['prepared_sha256']):
        raise ValueError('Prepared WM and A40 source are from different experiments')
    baseline = quick_check(source, args.baseline_panel, execution, NATIVE_SEED)
    if read(args.baseline_panel/'metrics.json')['actor_state_sha256'] != source_identity['actor_sha256']:
        raise ValueError('Native baseline does not match A40')
    baseline_file = output/'search_state.json'
    generator = np.random.default_rng(SEARCH_SEED)
    case_ids = np.random.default_rng(SEARCH_SEED).permutation(np.arange(1, 128, 2)).tolist()
    if args.resume:
        saved = read(baseline_file)
        if saved['source'] != source_identity or saved['cases'] != case_ids:
            raise ValueError('Cannot resume a different native search')
        generator.bit_generator.state = saved['rng']
        history = saved['history']
        scale, center, incumbent = saved['scale'], np.array(saved['center']), saved['incumbent']
    else:
        output.mkdir()
        history, scale, center, incumbent = [], .004, np.zeros(6), [0.] * 6

    def save_state():
        dump(baseline_file, dict(source=source_identity, cases=case_ids, history=history,
                                 scale=scale, center=center.tolist(), incumbent=incumbent,
                                 rng=generator.bit_generator.state))

    save_state()
    status = output/'status.json'
    screen = fresh_path(output/'wm_screen')
    process = None
    log = None
    if len(history) < GENERATIONS:
        cache = output.parent / '.native_replay_cache'
        if cache.is_dir() and not (output / '.native_replay_cache').exists():
            (output / '.native_replay_cache').symlink_to(cache.resolve(), target_is_directory=True)
        command = train_command(registered, prepared, screen, True, stop_at=48, fresh=False,
                                prioritize_flight=True, start_lanes=32, guard_lanes=16, e1_v2=True,
                                labeled_start_checkpoint=args.history_checkpoint, labeled_snapshot_reserve=8)
        command.extend(('--approach-native-search-screen', '--approach-directed-risk',
                        '--wm-action-collection', str(args.directed_collection.resolve(strict=True)),
                        '--approach-speed-risk-history', str(args.risk_history.resolve(strict=True))))
        for lane in (10, 28, 52, 78):
            command.extend(('--approach-directed-validation-lane', str(lane)))
        dump(status, dict(status='loading_frozen_wm', completed_generations=len(history)))
        log = screen.with_suffix('.log').open('ab')
        process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT,
                                   env=dict(os.environ, PYTHONPATH='.:scripts', QUADSMASH_ZSTD_ARCHIVE='1'),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    serial = 1
    try:
        screen_identity = wait_json(screen/'screen_ready.json', process) if process else None
        if screen_identity and (screen_identity['actor_sha256'] != source_identity['actor_sha256']
                or screen_identity['world_sha256'] != preparation['world_sha256']
                or screen_identity['guard_lanes'] != 16):
            raise ValueError('Frozen WM screener initialized with a different experiment identity')
        for generation in range(len(history) + 1, GENERATIONS + 1):
            dump(status, dict(status='screening', generation=generation, completed_generations=len(history)))
            screened = []
            for _ in range(3):
                offsets = proposals(incumbent, center, scale, generator)
                dump(screen/f'screen_request_{serial:03d}.json', dict(offsets=offsets))
                report = wait_json(screen/f'screen_result_{serial:03d}.json', process)
                serial += 1
                if any(report[k] != screen_identity[k] for k in screen_identity):
                    raise ValueError('Frozen WM screener changed identity')
                shortlisted = [row['offset'] for row in report['candidates'] if row['passed']][:4]
                if not shortlisted or shortlisted[0] != incumbent:
                    raise ValueError('Screening lost the measured incumbent')
                screened.append(report)
                if len(shortlisted) >= 2:
                    break
                center = np.array(incumbent)
                scale = max(.000125, scale / 2.)
            else:
                raise RuntimeError('Frozen WM found no distinct admissible native challenger after three draws')

            offset = (generation - 1) % 2 * 32
            cases = case_ids[offset:offset+32]
            directory = fresh_path(output/f'generation_{generation:02d}')
            spec = output/f'{directory.name}_spec.json'
            dump(spec, dict(offsets=shortlisted, case_indices=cases))
            command = evaluate_command(source, directory, True, execution, seed=NATIVE_SEED, a0=False)
            command[command.index('--research-fixed128')] = '--approach-search-spec'
            command.append(str(spec))
            dump(status, dict(status='native_generation', generation=generation, completed_generations=len(history),
                              native_lanes=32*len(shortlisted), evaluation=str(directory)))
            run(command, directory.with_suffix('.log'), 40)
            result = read(directory/'approach_search_results.json')
            if (result['checkpoint_sha256'] != source_identity['checkpoint_sha256']
                    or result['actor_sha256'] != source_identity['actor_sha256']
                    or result['offsets'] != shortlisted or result['case_indices'] != cases
                    or read(directory/'completed.json')['episodes'] != 32*len(shortlisted)
                    or [row['offset'] for row in result['variants']] != shortlisted):
                raise ValueError('Native generation receipt changed source, proposals or cases')
            ranked = sorted(result['variants'], key=rank_key, reverse=True)
            best = ranked[0]
            previous = incumbent
            incumbent = best['offset']
            elite = np.asarray([row['offset'] for row in ranked[:2]])
            center = .25 * center + .75 * elite.mean(axis=0)
            scale = float(np.clip(.65 * scale + .35 * np.sqrt(np.mean(elite.var(axis=0))), .00035, .008))
            history.append(dict(generation=generation, native_lanes=result['native_lanes'],
                                case_indices=cases, incumbent_before=previous, incumbent_after=incumbent,
                                actor_changed=previous != incumbent,
                                baseline_success=result['variants'][0]['success'], selected_success=best['success'],
                                screening_seconds=sum(row['seconds'] for row in screened),
                                native_seconds=result['elapsed_s'], evaluation=str(directory.relative_to(output))))
            save_state()
    finally:
        if process:
            (screen/'screen_stop').touch()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()

    if len(history) != GENERATIONS:
        raise RuntimeError('Incomplete native A8 search')
    checkpoint = output/'approach_native_CEM_A08.pt'
    new_actor = actor_with_bias(original['actor'], incumbent)
    actor_sha = tensor_digest(new_actor.items())
    if not checkpoint.is_file():
        atomic_torch_save(dict(algorithm='supermimic_v1', actor=new_actor,
                               metadata=original['metadata'], adapter_schema=original['adapter_schema'],
                               native_approach_search=dict(source=source_identity, generations=GENERATIONS,
                                                           arm_bias=incumbent, optimizer='CEM')), checkpoint)
    else:
        exported = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if tensor_digest(exported['actor'].items()) != actor_sha:
            raise ValueError('Saved A8 evaluation policy is from a different search')
    panel = output/'A08_final_quick64'
    if panel.exists() and not (panel/'completed.json').exists():
        panel = fresh_path(panel)
    dump(status, dict(status='final_paired_evaluation', completed_generations=GENERATIONS,
                      actor_changes=sum(row['actor_changed'] for row in history), checkpoint=str(checkpoint)))
    evaluated = quick_check(checkpoint, panel, execution, NATIVE_SEED)
    if read(panel/'metrics.json')['actor_state_sha256'] != actor_sha:
        raise ValueError('Final panel did not evaluate the saved A8 policy')
    paired = paired_quick_result(baseline, evaluated, args.baseline_panel, panel)
    selected = checkpoint if evaluated['success'] > baseline['success'] else source
    dump(status, dict(status='completed', completed_generations=GENERATIONS,
                      actor_changes=sum(row['actor_changed'] for row in history), arm_bias=incumbent,
                      checkpoint_sha256=file_sha(checkpoint), selected_checkpoint=str(selected),
                      baseline=baseline, candidate=evaluated, paired=paired, generations=history,
                      final_evaluation=str(panel.relative_to(output))))


if __name__ == '__main__':
    main()
