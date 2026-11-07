"""Refine sustained support using fine dynamics for every selection decision.

Coarse feasibility is only an initializer. Every candidate starts at the same
recorded airborne state and predicts to the original episode boundary. This
does not run native physics or generate approved expert demonstrations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.search_landing_support_tail_v3 import initialize_worker, evaluate_job
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--population', type=int, default=12)
    parser.add_argument('--generations', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-seconds', type=float, default=900.)
    parser.add_argument('--seed', type=int, default=882113)
    args = parser.parse_args()
    if (args.output.exists() or not 4 <= args.population <= 32 or
            not 1 <= args.generations <= 6 or not 1 <= args.workers <= 4 or
            not 120 <= args.max_seconds <= 1800):
        raise ValueError('Fresh output and bounded fine search required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = read(args.source)
    hashes = dict(source['source_sha256'])
    hashes[str(args.source.resolve())] = sha(args.source)
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    for filename, expected in hashes.items():
        if sha(filename) != expected:
            raise ValueError('Changed source ' + filename)
    parent = read(source['configuration']['source_search'])
    configuration = parent['configuration']
    query = read(configuration['query'])
    manifest = read(Path(configuration['bundle']) / 'manifest.json')
    if source['case'] != 3 or source['tick'] != 280:
        raise ValueError('Verified ground-start query required')
    initargs = (manifest['configuration'], query,
                source['fixed_early_knots_rad'], configuration['sphere_audit'])
    # These are initialization guesses, not fine successes. Include the two
    # previous fine failures and each distinct coarse feasible candidate.
    seeds = []
    seen = set()
    candidates = sorted(source['fine_candidates'], key=lambda x: x['fine']['rank'])
    candidates += [x for x in source['ledger'] if x['result']['accepted']]
    for row in candidates:
        knots = np.asarray(row['knots_rad'], float)
        key = knots.tobytes()
        if key not in seen:
            seen.add(key)
            seeds.append(knots)
    if not seeds:
        raise ValueError('No bounded initialization available')
    mean = seeds[0].copy()
    bound = np.asarray(source['tail_bounds_rad'])
    std = np.tile(np.r_[np.full(12, .045), np.full(6, .02)], (3, 1))
    floor = np.tile(np.r_[np.full(12, .012), np.full(6, .008)], (3, 1))
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    deadline = started + args.max_seconds
    search_deadline = started + .80 * args.max_seconds
    ledger = []
    best = None
    best_knots = mean.copy()
    selected_prefix = min(parent['fine_candidates'], key=lambda x: x['fine']['rank'])['fine']['trajectory'][:80]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_partial():
        path = args.output.with_suffix('.partial.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(ledger), allow_nan=False) + '\n')
        temporary.replace(path)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=initargs) as pool:
        for generation in range(args.generations):
            if time.monotonic() >= search_deadline:
                break
            population = np.clip(rng.normal(mean, std, (args.population, 3, 18)), -bound, bound)
            population[0] = best_knots
            if generation == 0:
                for index, knots in enumerate(seeds[:args.population]):
                    population[index] = knots
            ranked = []
            jobs = [(i, k, 160, search_deadline, False) for i, k in enumerate(population)]
            for index, knots, result in pool.map(evaluate_job, jobs, chunksize=1):
                ledger.append(dict(generation=generation, index=index, knots_rad=knots, result=result))
                # An expired computation is not a bad physical trajectory and
                # must not steer the distribution towards a shorter rollout.
                if (result.get('failure') or {}).get('reason') != 'declared_compute_budget':
                    ranked.append((result['rank'], knots))
                    if best is None or result['rank'] < best['rank']:
                        best = result
                        best_knots = knots.copy()
                        print(json.dumps(plain(dict(generation=generation, index=index, best=best))), flush=True)
                write_partial()
            if not ranked:
                break
            ranked.sort(key=lambda x: x[0])
            elite = np.stack([x[1] for x in ranked[:max(2, len(ranked)//4)]])
            mean = elite.mean(0)
            std = np.maximum(elite.std(0), floor)
        replay = None
        prefix_error = None
        reproducible = False
        if best is not None and time.monotonic() < deadline:
            _, _, replay = next(pool.map(evaluate_job, [(0, best_knots, 160, deadline, True)]))
            if len(replay['trajectory']) >= 80:
                prefix_error = max(float(np.max(abs(np.asarray(x[k])-np.asarray(y[k]))))
                    for x, y in zip(replay['trajectory'][:80], selected_prefix)
                    for k in ['target', 'q', 'v'])
                if prefix_error > 1e-12:
                    raise ValueError('Fine refinement changed the protected landing prefix')
            reproducible = all(replay[key] == best[key] for key in
                ['accepted', 'completed_model_steps', 'failure', 'terminal', 'rank'])
    for filename, expected in hashes.items():
        if sha(filename) != expected:
            raise ValueError('Changed input during fine search ' + filename)
    output = dict(scope=__doc__, case=3, tick=280,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        selection_substeps=160, candidate_count=len(ledger), ledger=ledger,
        fixed_early_knots_rad=source['fixed_early_knots_rad'], best_knots_rad=best_knots,
        best=best, independent_replay=replay, replay_reproduced=reproducible,
        preserved_prefix_maximum_difference=prefix_error,
        any_model_feasible=bool(reproducible and replay['accepted'] and prefix_error == 0.),
        wall_seconds=time.monotonic()-started, source_sha256=hashes,
        native_execution=False, valid_expert_labels=False,
        limitations=['Fine model acceptance is not native feasibility or proof of a robust recovery region.',
                     'Head collision response, self collisions and ball-side legality still require native validation.'])
    args.output.write_text(json.dumps(plain(output), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
