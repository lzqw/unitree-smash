"""Joint early landing/support refinement with executed model regression gates.

Optimize both early 3x18 knots and late 3x18 knots. A better nominal/difficult
scenario score is eligible only after all previously successful model probes
still complete the original episode. This is offline model search, not native
execution, an invariant region certificate, or usable positive teacher data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.search_landing_support_tail_v3 import initialize_worker, _WORKER, SupportTailReference
from scripts.landing_reference_tracking_v1 import LandingReferenceTrackingV1
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.perturb_early_support_domain_v1 import perturb_state
from scripts.refine_multistate_support_v1 import candidate_rank
from scripts.audit_joint_sequence_proposal_v1 import plain


def joint_factory(model, interface, initial, knots, *, reference_q, reference_v, gains):
    knots = np.asarray(knots, float)
    if knots.shape != (6, 18) or not np.isfinite(knots).all():
        raise ValueError('Six finite full-body knots required')
    base = SupportTailReference(model, interface, initial, knots[3:], fixed_knots=knots[:3])
    return LandingReferenceTrackingV1(base, reference_q, reference_v, **gains)


def initialize(configuration, query, early, spheres, reference_q, reference_v, gains):
    initialize_worker(configuration, query, early, spheres)
    _WORKER['factory'] = partial(joint_factory, reference_q=reference_q,
                                 reference_v=reference_v, gains=gains)


def run_probe(probe, knots, deadline, save):
    w = _WORKER
    state = perturb_state(w['initial'], probe)
    result = evaluate(w['controller'], state, knots, w['spheres'], w['geometry'],
                      w['protocol'], substeps=160, deadline=deadline, save=save,
                      policy_factory=w['factory'], control_tick=280)
    return dict(probe=probe, result=result)


def preservation_passes(rows, required_names):
    names = [r['probe']['name'] for r in rows]
    return (len(names) == len(required_names) and set(names) == set(required_names)
            and all(r['result']['accepted'] for r in rows))


def proposal_deltas(joint_order, seed):
    """Early flexion candidates plus small temporally correlated whole-body moves."""
    rng = np.random.default_rng(seed)
    calf = joint_order.index('FL_calf_joint')
    front = [calf, joint_order.index('FR_calf_joint')]
    proposals = []
    for amount in (.005, .015, .03):
        delta = np.zeros((6, 18))
        delta[:3, calf] = -amount * np.array([.25, .6, 1.])
        proposals.append(('early_left_calf_flexion_' + str(amount), delta))
    delta = np.zeros((6, 18))
    delta[:3, front] = -.01 * np.array([.25, .6, 1.])[:, None]
    proposals.append(('early_front_calves_flexion', delta))
    for index in range(3):
        shared = rng.normal(0., np.r_[np.full(12, .006), np.full(6, .003)], (2, 1, 18))
        independent = rng.normal(0., np.r_[np.full(12, .002), np.full(6, .001)], (2, 3, 18))
        delta = (shared + independent).reshape(6, 18)
        proposals.append(('coupled_early_and_tail_' + str(index), delta))
    return proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=882116)
    args = parser.parse_args()
    if args.output.exists() or not 1200 <= args.max_seconds <= 2400 or not 1 <= args.workers <= 4:
        raise ValueError('Fresh output and bounded search required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source = read(args.source)
    if not source['nominal_preserved'] or source['compute_expired_probes']:
        raise ValueError('Completed reproducible source required')
    fine_paths = [Path(p) for p in source['source_sha256'] if Path(p).name == 'refine_support_tail_fine160_v1.json']
    if len(fine_paths) != 1:
        raise ValueError('Unique nominal source required')
    fine = read(fine_paths[0]); tail = read(fine['configuration']['source'])
    parent = read(tail['configuration']['source_search']); cfg = parent['configuration']
    query = read(cfg['query']); manifest = read(Path(cfg['bundle']) / 'manifest.json')
    trajectory = fine['independent_replay']['trajectory']
    reference_q = np.asarray([query['initial_q']] + [r['q'] for r in trajectory[:-1]])
    reference_v = np.asarray([query['initial_v']] + [r['v'] for r in trajectory[:-1]])
    initargs = (manifest['configuration'], query, fine['fixed_early_knots_rad'],
                cfg['sphere_audit'], reference_q, reference_v, source['configuration'])
    initialize(*initargs)
    model = _WORKER['controller'].qp.model
    original = np.vstack([fine['fixed_early_knots_rad'], fine['best_knots_rad']])
    trust = np.tile(np.r_[np.full(12, .04), np.full(6, .02)], (6, 1))
    old = {r['probe']['name']: r for r in source['rows']}
    fit_names = ['nominal', 'orientation_world_0_+1', 'orientation_world_1_+1']
    preserved_names = [n for n, r in old.items() if r['result']['accepted']]
    if len(preserved_names) != 5 or 'nominal' not in preserved_names:
        raise ValueError('The declared five-model-success baseline is required')
    extra_names = [n for n in preserved_names if n != 'nominal']
    hashes = dict(source['source_sha256'])
    for p in [args.source, Path(__file__), ROOT / 'scripts/refine_multistate_support_v1.py']:
        hashes[str(p.resolve())] = sha(p)
    for p, digest in hashes.items():
        if sha(p) != digest:
            raise ValueError('Changed source ' + p)
    start = time.monotonic(); deadline = start + args.max_seconds
    search_deadline = start + .70 * args.max_seconds
    best = original.copy(); best_rank = candidate_rank([old[n]['result'] for n in fit_names])
    ledger = []; baseline_reproduced = False; final = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint():
        value = dict(ledger=ledger, best_rank=best_rank, best_knots=best,
                     preserved_names=preserved_names, native_execution=False)
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(value), allow_nan=False) + '\n')
        tmp.replace(args.output.with_suffix('.partial.json'))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=initargs) as pool:
        def batch(names, knots, limit, save=False):
            jobs = [pool.submit(run_probe, old[n]['probe'], knots, limit, save) for n in names]
            lookup = {r['probe']['name']: r for f in as_completed(jobs) for r in [f.result()]}
            return [lookup[n] for n in names]
        for label, delta in [('baseline', np.zeros_like(original))] + proposal_deltas(model.joint_order, args.seed):
            if time.monotonic() >= search_deadline:
                break
            knots = np.clip(best + delta, original - trust, original + trust)
            rows = batch(fit_names, knots, search_deadline)
            rank = candidate_rank([r['result'] for r in rows])
            protected = []; eligible = False
            if label == 'baseline':
                baseline_reproduced = all(all(r['result'][k] == old[r['probe']['name']]['result'][k]
                    for k in ['accepted', 'completed_model_steps', 'failure', 'terminal', 'rank']) for r in rows)
                if not baseline_reproduced:
                    raise ValueError('Joint parameterization changed the original baseline')
            elif rank is not None and rank[0] == 0 and rank < best_rank:
                protected = batch(extra_names, knots, search_deadline)
                eligible = preservation_passes([rows[0]] + protected, preserved_names)
                if eligible:
                    best = knots.copy(); best_rank = rank
            ledger.append(dict(label=label, knots_rad=knots, fit_rows=rows, fit_rank=rank,
                               preservation_rows=protected, accepted_into_search=eligible))
            print(json.dumps(plain(dict(label=label, fit_rank=rank, accepted_into_search=eligible,
                preserved_results={r['probe']['name']: r['result']['accepted'] for r in protected}))), flush=True)
            checkpoint()
        if time.monotonic() < deadline:
            final = batch(list(old), best, deadline, save=True)
    for p, digest in hashes.items():
        if sha(p) != digest:
            raise ValueError('Source changed during run ' + p)
    complete = len(final) == 13 and not any((r['result']['failure'] or {}).get('reason') == 'declared_compute_budget' for r in final)
    retained = complete and preservation_passes([r for r in final if r['probe']['name'] in preserved_names], preserved_names)
    new_names = [r['probe']['name'] for r in final if r['result']['accepted'] and not old[r['probe']['name']]['result']['accepted']]
    result = dict(scope=__doc__, configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source_sha256=hashes, baseline_reproduced=baseline_reproduced, preserved_names=preserved_names,
        fit_names=fit_names, ledger=ledger, best_knots_rad=best, best_rank=best_rank,
        final_checks=final, all_thirteen_completed=complete, original_five_retained=retained,
        new_model_successes=new_names, accepted_model_probes=sum(r['result']['accepted'] for r in final),
        model_candidate_improved=bool(retained and new_names), wall_seconds=time.monotonic()-start,
        native_execution=False, valid_expert_labels=False,
        limitations=['All scenarios are previously inspected development probes, not held-out validation.',
                     'Fixed gains and root reference may restrict early trajectory changes.',
                     'Finite bounded proposals cannot prove physical impossibility.',
                     'Model feasibility does not certify nonfoot impact response or self collision.'])
    args.output.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
