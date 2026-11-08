"""Joint early full-body/timing search from a measured failed family state.

Optimize 54 early leg/arm reference coordinates and four phase times together.
The existing sustained feedback/transition remains active in every full-tail
prediction. This does not change physical constraints or certify native success.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
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
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.search_recovery_timing_v1 import evaluate, bounded
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model-plan', 'warm-search', 'failed-local', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh evidence directory required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    failed, warm_source = read(args.failed_local), read(args.warm_search)
    if failed['accepted'] or not failed['query_is_native'] or failed['control_tick'] != 280:
        raise ValueError('Actually observed early failed recovery query required')
    setup = read(args.model_plan)
    source = read(setup['source'])
    *_, geometry, context = source_context(source['configuration']['source'])
    initialize(*context)
    context, _ = transfer_reference(context, state_from_row(warm_source['query']))
    context, alignment = transfer_reference(context, state_from_row(failed['query']))
    initialize(*context)
    state = state_from_row(failed['query'])
    interface = _WORKER['controller'].qp.interface
    warm = copy.deepcopy(failed['records'][0]['row']['plan'])
    if warm['corrections']:
        raise ValueError('No executed future correction may be invented')
    warm.update(contract='timed_foot_space_recovery_v1', early_timing_s=[.08, .20, .50, .04])
    original = np.asarray(warm['original_knots_rad'], float)
    lower = np.empty((3, 18)); upper = lower.copy()
    lower[:, :12] = np.maximum(interface.limits[:12, 0], original[:3, :12] - 1.)
    upper[:, :12] = np.minimum(interface.limits[:12, 1], original[:3, :12] + 1.)
    lower[:, 12:] = np.maximum(interface.limits[12:, 0] - state.joint_position[12:], -.6)
    upper[:, 12:] = np.minimum(interface.limits[12:, 1] - state.joint_position[12:], .6)
    mean = np.clip(original[:3], lower, upper)
    std = np.tile(np.r_[np.full(12, .25), np.full(6, .10)], (3, 1))
    floor = np.r_[np.full(12, .04), np.full(6, .02)]
    timing_mean = np.array(warm['early_timing_s'])
    timing_std = np.array([.03, .05, .12, .03])
    timing_floor = np.array([.008, .012, .025, .008])
    pins = dict(failed['source_sha256'])
    for path in [args.model_plan, args.warm_search, args.failed_local, Path(__file__)]:
        pins[str(path.resolve())] = sha(path)
    for name, module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module, '__file__', None):
            path = str(Path(module.__file__).resolve())
            digest = sha(path)
            if path in pins and pins[path] != digest:
                raise ValueError('Changed implementation ' + path)
            pins[path] = digest
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed source ' + path)
    args.output.mkdir(parents=True)

    def write(name, value):
        path = args.output / name; temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    write('owner.json', dict(pid=os.getpid()))
    write('inputs.json', dict(source_sha256=pins, case=failed['case'], workers=2,
        max_seconds=900, configuration={k: str(v) for k, v in vars(args).items()},
        early_lower=lower, early_upper=upper, original_constraints_changed=False))
    started = time.monotonic(); deadline = started + 900.; search_end = started + 720.
    rng = np.random.default_rng(882174); ledger = []; verified = None

    def candidate(early, timing):
        plan = copy.deepcopy(warm); knots = original.copy()
        knots[:3] = np.clip(early, lower, upper)
        plan.update(original_knots_rad=knots, early_timing_s=bounded(timing))
        return plan

    def usable(row):
        return (row['result']['failure'] or {}).get('reason') != 'declared_compute_budget'

    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'),
                                 initializer=initialize, initargs=context) as pool:
            def batch(plans, resolution, limit):
                return list(pool.map(evaluate, [dict(plan=p, query=failed['query'],
                    geometry=geometry, substeps=resolution, deadline=limit) for p in plans]))

            baseline = batch([warm], 160, deadline)[0]
            old = failed['records'][0]['row']['result']
            for key in ['accepted', 'completed_model_steps', 'failure', 'rank', 'terminal']:
                if baseline['result'][key] != old[key]:
                    raise ValueError('Default timing changed original warm prediction: ' + key)
            write('baseline.json', baseline); best = baseline
            for generation in range(6):
                if time.monotonic() >= search_end:
                    break
                plans = [candidate(mean + std * (.75 * rng.normal(size=(1, 18))
                    + .5 * rng.normal(size=(3, 18))), timing_mean + timing_std * rng.normal(size=4))
                    for _ in range(48)]
                plans[:3] = [copy.deepcopy(warm), candidate(mean, timing_mean), copy.deepcopy(best['plan'])]
                if generation == 0:
                    # Deterministic candidates couple earlier leg deployment
                    # with measured-arm braking, not fabricated ground forces.
                    measured = np.tile(np.r_[state.joint_position[:12], np.zeros(6)], (3, 1))
                    plans[3] = candidate(measured, [.04, .10, .25, .04])
                    plans[4] = candidate(original[:3], [.04, .10, .25, .04])
                    plans[5] = candidate(original[:3], [.06, .14, .35, .10])
                coarse = batch(plans, 32, search_end)
                ranked = sorted([r for r in coarse if usable(r)], key=lambda r: r['result']['rank'])
                if not ranked:
                    break
                elite = ranked[:8]
                e = np.stack([r['plan']['original_knots_rad'][:3] for r in elite])
                t = np.stack([r['plan']['early_timing_s'] for r in elite])
                mean = .3 * mean + .7 * e.mean(0)
                std = np.maximum(floor, .3 * std + .7 * e.std(0))
                timing_mean = bounded(.3 * timing_mean + .7 * t.mean(0))
                timing_std = np.maximum(timing_floor, .3 * timing_std + .7 * t.std(0))
                fine = batch([r['plan'] for r in ranked[:4]], 160, deadline)
                for row in fine:
                    if usable(row) and row['result']['rank'] < best['result']['rank']:
                        best = row
                write(f'generation_{generation:02d}.json', dict(coarse=coarse, fine=fine))
                item = dict(generation=generation, completed=best['result']['completed_model_steps'],
                    timing=best['plan']['early_timing_s'], failure=best['result']['failure'])
                ledger.append(item); write('status.json', dict(state='SEARCHING', **item))
                print(json.dumps(plain(item)), flush=True)
                if best['result']['accepted']:
                    replay = batch([best['plan']], 160, deadline)[0]
                    a, b = best['result'], replay['result']
                    if b['accepted'] and len(a['trajectory']) == len(b['trajectory']) == 520 and all(
                            np.array_equal(x[k], y[k]) for x, y in zip(a['trajectory'], b['trajectory'])
                            for k in ['q', 'v', 'target', 'contacts']):
                        verified = replay
                    break
        for path, digest in pins.items():
            if sha(path) != digest:
                raise ValueError('Source changed during search ' + path)
        write('result.json', dict(scope=__doc__, case=failed['case'], query=failed['query'],
            reference_alignment=alignment, baseline_reproduced=True, best=best, verified=verified,
            ledger=ledger, source_sha256=pins, wall_seconds=time.monotonic()-started,
            native_execution=False, valid_expert_labels=False, physical_impossibility_claim=False))
        write('status.json', dict(state='COMPLETED', model_accepted=verified is not None,
            completed=best['result']['completed_model_steps']))
    except BaseException as error:
        write('status.json', dict(state='FAILED', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
