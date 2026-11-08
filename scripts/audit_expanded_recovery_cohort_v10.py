"""Fine-plant full-episode feedback test for newly added model references."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_recovery_long_reference_fine_v8 import evaluate
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'fixture-run', 'bank', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    args = parser.parse_args()
    if args.output.exists() or not 120 <= args.max_seconds <= 2400:
        raise ValueError('Fresh bounded model evaluation required')
    read = lambda p: json.loads(Path(p).read_text())
    cfg = read(args.configuration)
    nominal = read(args.fixture_run / 'native_command_inputs.json')['default_joint_position_rad']
    library = read(cfg['model_reference_library'])
    bank = {r['case']: r for r in read(args.bank)['queries']}
    selected = set(library['added_source_cases'])
    indices = [i for i, r in enumerate(library['references']) if r['source_case'] in selected]
    if not indices or len(indices) != len(selected):
        raise ValueError('New cohort provenance mismatch')
    args.output.mkdir(parents=True)

    def write(name, value):
        path = args.output / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    files = [Path(__file__), args.configuration, args.bank,
             args.fixture_run / 'native_command_inputs.json', Path(cfg['model_reference_library'])]
    files += [Path(module.__file__) for name, module in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(module, '__file__', None)]
    pins = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write('inputs.json', dict(source_sha256=pins, selected_cases=sorted(selected), workers=6,
        max_seconds=args.max_seconds, plant_substeps=160, controller_prediction_substeps=16,
        original_episode_ticks=800, training=False, native_success=False))
    started = time.monotonic()
    results = []

    def status(state):
        write('status.json', dict(state=state, pid=os.getpid(), completed=len(results), total=len(indices)))

    status('RUNNING_FINE_MODEL_FULL_EPISODES')
    try:
        with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(evaluate, cfg, nominal, i, 'predictive_selection',
                started + args.max_seconds,
                800 - int(bank[library['references'][i]['source_case']]['control_tick'])) for i in indices]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                write('case_%03d.json' % result['source_case'], result)
                status('RUNNING_FINE_MODEL_FULL_EPISODES')
        for name, expected in pins.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Changed model evaluation input ' + name)
        summary = dict(state='COMPLETED_FINE_MODEL_FULL_EPISODES', cases=len(results),
            model_episode_end_cases=sorted(r['source_case'] for r in results if r['steps'] == r['required_steps']),
            model_recovery_cases=sorted(r['source_case'] for r in results if r['steps'] == r['required_steps'] and r['final_model_stable_s'] >= .8),
            wall_seconds=time.monotonic() - started, native_success=False, valid_expert_labels=False)
        write('result.json', summary)
        write('status.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write('status.json', dict(state='FAILED', pid=os.getpid(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
