"""Search uncovered full18 landing states around actual elite trajectories.

Common candidate construction and objective for every state. Existing model
solutions supply warm starts, not expert labels or a native success claim.
All initial states are recorded causal release states; native physics is not
reset or teleported. Full episode validation is a separate required step.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import search_full_valid_ball_recovery_v2 as search
from scripts.audit_joint_sequence_proposal_v1 import plain


def initialize(configuration, nominal, sources):
    from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
    from scripts.whole_body_support_qp_v1 import PositionPDInterface
    search.WORK['controller'] = PureControlReferenceClockV12(
        PositionPDInterface(**configuration['interface']), nominal,
        configuration['model_reference_library'])
    # Reuse only the predictor and initial-state constructor. No matching or
    # controller.command call is made by the offline rollout evaluator.
    search.WORK['controller'].library.records = sources


def structured_noise(rng, sigma, spans):
    physical = rng.normal(0, .25 * sigma, (24, 8, 18))
    extension = rng.normal(0, sigma, (24, 8, 1)) + rng.normal(0, .4 * sigma, (24, 8, 4))
    physical[:, :, 8:12] += extension
    physical[:, :, 4:8] -= .5 * extension
    lateral = rng.normal(0, .5 * sigma, (24, 8, 1))
    spread = rng.normal(0, .5 * sigma, (24, 8, 1)) * np.array([1, -1, 1, -1])
    physical[:, :, :4] += lateral + spread
    physical[:, :, 12:] += rng.normal(0, sigma, (24, 8, 6))
    physical[:, 1:] = .5 * physical[:, 1:] + .5 * physical[:, :-1]
    return 2 * physical / spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'fixture-run', 'bank', 'baseline-metrics',
                 'warm-search', 'verified-model-results', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=3600.)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    if args.output.exists() or not 900 <= args.max_seconds <= 5400:
        raise ValueError('Fresh bounded cohort search required')
    cfg = read(args.configuration)
    nominal = np.asarray(read(args.fixture_run / 'native_command_inputs.json')['default_joint_position_rad'])
    bank = {r['case']: r for r in read(args.bank)['queries']}
    baseline = read(args.baseline_metrics)['complete5_case_outcomes']
    cohort = sorted(r['training_index'] for r in baseline if r['ball5_except_recovery'] and not r['recovery'])
    verified_paths = sorted(args.verified_model_results.glob('case_*.json'))
    previous = {r['source_case'] for p in verified_paths for r in [read(p)]
                if r['failure'] is None and r['steps'] == r['required_steps'] and r['final_model_stable_s'] >= .8}
    selected = [case for case in cohort if case not in previous]
    if not selected or set(selected) - set(bank):
        raise ValueError('Expected remaining actual release states')
    sources = []
    for case in selected:
        query = bank[case]['query']
        for name, width in [('q', 25), ('v', 24), ('target', 18), ('contacts', 4)]:
            if np.shape(query[name]) != (width,) or not np.isfinite(query[name]).all():
                raise ValueError('Malformed recorded state')
        key = hashlib.sha256(json.dumps(query, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        sources.append(dict(source_case=case, source_key=key, query=query))
    initialize(cfg, nominal, sources)
    controller = search.WORK['controller']
    limits = controller.interface.limits
    spans = np.diff(limits).ravel()
    warm_paths = [p for p in sorted(args.warm_search.glob('case_*.json')) if read(p)['source_case'] in previous]
    warm = [read(p) for p in warm_paths]
    if not warm or any(not r['fine']['model_path_complete'] for r in warm):
        raise ValueError('Fine-checked model warm starts required')
    args.output.mkdir(parents=True)

    def write(name, value):
        path = args.output / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    pins = [Path(__file__), Path(search.__file__), args.configuration, args.bank, args.baseline_metrics,
            args.fixture_run / 'native_command_inputs.json', Path(cfg['model_reference_library']), *warm_paths, *verified_paths]
    pins += [Path(m.__file__) for n,m in list(sys.modules.items())
             if n.startswith('scripts.') and getattr(m,'__file__',None)]
    source_pins = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in pins}
    write('inputs.json', dict(source_sha256=source_pins,
        sampling='around actual elites; no mean across distinct contact sequences', fine_candidates_per_case=3,
        full_ball_valid_failure_cohort=cohort, already_full_model_candidates=sorted(previous & set(cohort)),
        selected_cases=selected, selection='Remaining baseline ball5_except_recovery and not recovery, without a verified full-model candidate',
        optimized_horizon_s=1.6, command_knots=8, command_dimensions=18, workers=8,
        max_seconds=args.max_seconds, common_warm_start_cases=[r['source_case'] for r in warm],
        native_state_teleport=False, native_success=False, training=False, valid_expert_labels=False))
    times = np.asarray([.08, .18, .30, .45, .60, .90, 1.20, 1.60])
    normalize = lambda targets: np.clip(2 * (targets - limits[:, 0]) / spans - 1, -1, 1)
    seeds = []
    for source in sources:
        actual = np.asarray(source['query']['target'])
        proposals = [normalize(np.tile(nominal, (8, 1)))]
        for item in warm:
            targets = limits[:, 0] + .5 * (np.asarray(item['fine']['nodes']) + 1) * spans
            proposals.append(normalize(targets))
            difference = actual - np.asarray(bank[item['source_case']]['query']['target'])
            proposals.append(normalize(targets + np.exp(-times[:, None] / .3) * difference))
        seeds.append(proposals)
    started = time.monotonic()
    deadline = started + args.max_seconds
    coarse = [[] for _ in sources]
    completed = 0

    def status(stage, **extra):
        write('status.json', dict(state=stage, pid=os.getpid(), cases=len(sources),
            completed_candidates=completed, elapsed_s=time.monotonic() - started, **extra))

    try:
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'),
                                 initializer=initialize, initargs=(cfg, nominal, sources)) as pool:
            batches = []
            for seed_index in range(len(seeds[0])):
                batches.extend((i, seeds[i][seed_index]) for i in range(len(sources)))
            for generation in range(4):
                if generation:
                    if time.monotonic() >= deadline - 600:
                        break
                    batches = []
                    sigma = [.30, .20, .10][generation - 1]
                    for i in range(len(sources)):
                        coarse[i].sort(key=lambda r: r['rank'])
                        elite = coarse[i][:min(5, len(coarse[i]))]
                        rng = np.random.default_rng(885000 + 100 * generation + sources[i]['source_case'])
                        batches.extend((i, np.clip(np.asarray(elite[j % len(elite)]['nodes']) + n, -1, 1))
                            for j, n in enumerate(structured_noise(rng, sigma, spans)))
                status('SEARCHING_FULL_COMMAND_COHORT', generation=generation)
                futures = {pool.submit(search.rollout, i, nodes, 16, deadline - 600): i for i, nodes in batches}
                for future in as_completed(futures):
                    i = futures[future]
                    coarse[i].append(future.result())
                    completed += 1
                    if completed % 32 == 0:
                        status('SEARCHING_FULL_COMMAND_COHORT', generation=generation)
                for i, source in enumerate(sources):
                    coarse[i].sort(key=lambda r: r['rank'])
                    write('coarse_%03d.json' % source['source_case'], dict(source_case=source['source_case'],
                        source_key=source['source_key'], generation=generation, candidates=coarse[i]))
            status('FINE_VERIFYING_COHORT')
            futures = {pool.submit(search.rollout, i, np.asarray(candidate['nodes']), 160, deadline): (i, rank)
                       for i in range(len(sources)) for rank, candidate in enumerate(coarse[i][:3])}
            fine_results = [[] for _ in sources]
            for future in as_completed(futures):
                i, rank = futures[future]
                fine_results[i].append(dict(coarse_rank=rank, result=future.result()))
                status('FINE_VERIFYING_COHORT', fine_completed=sum(map(len, fine_results)))
            results = []
            for i, source in enumerate(sources):
                if not fine_results[i]:raise ValueError('Missing fine result')
                fine = min(fine_results[i], key=lambda r:r['result']['rank'])['result']
                result = dict(source_case=source['source_case'], source_key=source['source_key'],
                    query=source['query'], coarse_candidates=coarse[i], fine=fine,
                    fine_alternatives=[dict(coarse_rank=r['coarse_rank'],
                        **{k:v for k,v in r['result'].items() if k not in ['trajectory','terminal']})
                        for r in fine_results[i]],
                    full_episode_certified=False, native_execution=False, valid_expert_labels=False)
                write('case_%03d.json' % source['source_case'], result)
                results.append(result)
        for name, expected in source_pins.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Changed input during search: ' + name)
        summary = dict(state='COMPLETED_COHORT_SEARCH', cases=len(results), wall_seconds=time.monotonic() - started,
            fine_complete_cases=sorted(r['source_case'] for r in results if r['fine']['model_path_complete']),
            fine_stable_08s_cases=sorted(r['source_case'] for r in results if r['fine']['model_path_complete'] and r['fine']['terminal_stable_s'] >= .8),
            compute_budget_cases=sorted(r['source_case'] for r in results if (r['fine']['failure'] or {}).get('reason') == 'compute_budget'),
            native_success=False, full_episode_certified=False, valid_expert_labels=False)
        write('result.json', summary)
        write('status.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write('status.json', dict(state='FAILED', pid=os.getpid(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
