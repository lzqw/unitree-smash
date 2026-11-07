"""Refine early recovery commands without relaxing any physical acceptance test.

Nonuniform knots add resolution around the first 40 ms clearance violation.
The existing whole-body rollout and all its original checks remain unchanged.
This offline search does not execute a robot or generate verified labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.search_frictional_recovery_cem_v1 as core

KNOT_TICKS = np.array([0., 2., 5., 12., 25., 39.])
BASE = None


def expand_corrections(knots):
    knots = np.asarray(knots, float)
    if knots.shape != (6, 18) or not np.isfinite(knots).all():
        raise ValueError('Expected finite six-knot correction')
    return np.column_stack([
        np.interp(np.arange(40), KNOT_TICKS, knots[:, j]) for j in range(18)
    ])


def initialize(bundle, sphere_audit, prior_result):
    global BASE
    prior = json.loads(Path(prior_result).read_text())
    core.init_worker(bundle, sphere_audit, prior['case'])
    old = np.asarray(prior['knots'], float)
    if old.shape != (3, 18):
        raise ValueError('Expected existing three-knot search')
    BASE = np.column_stack([
        np.interp(np.arange(40), np.linspace(0, 39, 3), old[:, j])
        for j in range(18)
    ])


def evaluate(knots, save_states=False):
    return core.evaluate(BASE + expand_corrections(knots), save_states)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle', 'sphere-audit', 'start-result', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--seed', type=int, default=882097)
    a = p.parse_args()
    if a.output.exists() or not 16 <= a.population <= 128 or not 1 <= a.iterations <= 12:
        raise ValueError('Preserve evidence and bound search')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    prior = json.loads(a.start_result.read_text())
    hashes = dict(prior['source_sha256'])
    for f, digest in hashes.items():
        if sha(f) != digest:
            raise ValueError('Changed prior input ' + f)
    for f in [a.start_result, Path(__file__)]:
        hashes[str(f.resolve())] = sha(f)
    if str((a.bundle/'manifest.json').resolve()) not in hashes or str(a.sphere_audit.resolve()) not in hashes:
        raise ValueError('Unbound bundle or sphere evidence')
    mean = np.zeros((6, 18))
    std = np.tile(np.r_[np.full(12, .012), np.full(6, .006)], (6, 1))
    best_knots = mean.copy()
    rng = np.random.default_rng(a.seed)
    history = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize, initargs=(str(a.bundle), str(a.sphere_audit), str(a.start_result))) as pool:
        baseline = pool.submit(evaluate, mean, True).result()
        if not np.allclose(baseline['commands'], prior['final']['commands'], atol=1e-14, rtol=0):
            raise ValueError('Zero correction does not preserve prior commands')
        if abs(baseline['maximum_scaled_violation'] - prior['final']['maximum_scaled_violation']) > 1e-10:
            raise ValueError('Prior constraint check not reproduced')
        best = {k: v for k, v in baseline.items() if k not in ['states', 'commands']}
        best.update(states=None, commands=None)
        for iteration in range(a.iterations):
            candidates = mean + rng.normal(size=(a.population, 6, 18)) * std
            candidates[0] = best_knots
            values = list(pool.map(evaluate, candidates))
            order = sorted(range(len(values)), key=lambda i: core.key(values[i]))
            elite = candidates[order[:max(6, a.population // 6)]]
            mean = .2 * mean + .8 * elite.mean(0)
            std = np.maximum(.0003, .3 * std + .7 * elite.std(0))
            if core.key(values[order[0]]) < core.key(best):
                best = values[order[0]]
                best_knots = candidates[order[0]].copy()
            row = dict(iteration=iteration, best=best,
                accepted=sum(v['accepted'] for v in values),
                solver_failures=sum(not v['solved'] for v in values))
            history.append(row)
            print(json.dumps(row), flush=True)
            partial = dict(history=history, best_knots=best_knots.tolist(), mean=mean.tolist(),
                std=std.tolist(), random_state=rng.bit_generator.state, source_sha256=hashes,
                native_execution=False, valid_teacher=False)
            temp = a.output.with_suffix('.tmp')
            temp.write_text(json.dumps(partial, indent=2, allow_nan=False) + '\n')
            temp.replace(a.output.with_suffix('.partial.json'))
        final = pool.submit(evaluate, best_knots, True).result()
    if core.key(final) != core.key(best):
        raise ValueError('Independent replay differs')
    for f, digest in hashes.items():
        if sha(f) != digest:
            raise ValueError('Search input changed ' + f)
    result = dict(scope=__doc__, native_execution=False, valid_teacher=False,
        case=prior['case'], seed=a.seed, population=a.population, iterations=a.iterations,
        knot_ticks=KNOT_TICKS.tolist(), baseline=baseline, history=history,
        final=final, correction_knots=best_knots.tolist(),
        wall_seconds=time.perf_counter()-started, source_sha256=hashes,
        limitation='Only 0.4 s model search; requires sustained feedback and native validation')
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
