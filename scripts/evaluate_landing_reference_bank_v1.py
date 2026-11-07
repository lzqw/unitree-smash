"""Full-episode prediction-based selection among complementary recovery references.

At the same measured model state, predict every candidate with the unchanged
fine coupled dynamics. Keep the baseline when feasible; otherwise select only
a fully feasible alternative. No native outcome or scenario ID enters selection.
This tests the initial planning layer, not a deployed receding-horizon MPC.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.perturb_early_support_domain_v1 import perturb_state
from scripts.search_early_landing_trajectory_v1 import evaluate
from scripts.audit_joint_sequence_proposal_v1 import plain


def state_digest(state):
    digest = hashlib.sha256()
    for value in [state.q, state.v, state.contacts, state.previous_command_target]:
        digest.update(np.asarray(value, dtype='<f8').tobytes())
    return digest.hexdigest()


def select_prediction(rows):
    if not rows or len({r['state_sha256'] for r in rows}) != 1:
        raise ValueError('Predictions must use the same current state and command history')
    names = [r['candidate'] for r in rows]
    if len(names) != len(set(names)) or 'baseline' not in names:
        raise ValueError('Unique candidates including baseline are required')
    feasible = []
    for row in rows:
        result = row['result']
        if not result['accepted']:
            continue
        if (result['required_model_steps'] != 520 or result['completed_model_steps'] != 520
                or result['failure'] is not None or result['terminal_violations']
                or result['trailing_model_window_s'] < .8 or not np.isfinite(result['cost'])):
            raise ValueError('Claimed feasibility lacks complete constrained continuation')
        feasible.append(row)
    baseline = next((r for r in feasible if r['candidate'] == 'baseline'), None)
    if baseline is not None:
        return baseline
    return min(feasible, key=lambda r: (r['result']['cost'], r['candidate'])) if feasible else None


def job(probe, candidate, knots, deadline):
    w = _WORKER
    state = perturb_state(w['initial'], probe)
    result = evaluate(w['controller'], state, knots, w['spheres'], w['geometry'], w['protocol'],
        substeps=160, deadline=deadline, save=True, policy_factory=w['factory'], control_tick=280)
    return dict(probe=probe, candidate=candidate, state_sha256=state_digest(state), result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not 900 <= args.max_seconds <= 2400 or not 1 <= args.workers <= 4:
        raise ValueError('Fresh result and finite computation budget required')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    search = read(args.search)
    if not search['baseline_reproduced'] or not search['all_thirteen_completed']:
        raise ValueError('Completed reproducible precursor search required')
    source_path = Path(search['configuration']['source']); source = read(source_path)
    fine_paths = [Path(p) for p in source['source_sha256'] if Path(p).name == 'refine_support_tail_fine160_v1.json']
    if len(fine_paths) != 1: raise ValueError('Unique nominal reference required')
    fine = read(fine_paths[0]); tail = read(fine['configuration']['source'])
    parent = read(tail['configuration']['source_search']); cfg = parent['configuration']
    query = read(cfg['query']); manifest = read(Path(cfg['bundle'])/'manifest.json')
    states = fine['independent_replay']['trajectory']
    rq = np.asarray([query['initial_q']] + [r['q'] for r in states[:-1]])
    rv = np.asarray([query['initial_v']] + [r['v'] for r in states[:-1]])
    initargs = (manifest['configuration'], query, fine['fixed_early_knots_rad'],
                cfg['sphere_audit'], rq, rv, source['configuration'])
    baseline = np.vstack([fine['fixed_early_knots_rad'], fine['best_knots_rad']])
    complementary = [r for r in search['ledger'] if r['label'] == 'early_front_calves_flexion']
    if len(complementary) != 1 or not any(r['result']['accepted'] for r in complementary[0]['fit_rows'][1:]):
        raise ValueError('Recorded complementary full-episode solution required')
    candidates = [('baseline', baseline), ('preserved_refinement', np.asarray(search['best_knots_rad'])),
                  ('front_leg_alternative', np.asarray(complementary[0]['knots_rad']))]
    bank = []; seen = set()
    for name, knots in candidates:
        if knots.shape != (6, 18) or not np.isfinite(knots).all(): raise ValueError('Malformed reference')
        identity = knots.tobytes()
        if identity not in seen:
            bank.append((name, knots)); seen.add(identity)
    hashes = dict(search['source_sha256'])
    for path in [args.search, Path(__file__)]: hashes[str(path.resolve())] = sha(path)
    for path, digest in hashes.items():
        if sha(path) != digest: raise ValueError('Changed source ' + path)
    started = time.monotonic(); deadline = started + args.max_seconds
    rows = []; selections = []; groups = {}; old = {r['probe']['name']: r for r in source['rows']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint():
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(plain(dict(selections=selections, completed_predictions=len(rows))), allow_nan=False)+'\n')
        temp.replace(args.output.with_suffix('.partial.json'))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=initargs) as pool:
        futures = [pool.submit(job, probe['probe'], name, knots, deadline)
                   for probe in source['rows'] for name, knots in bank]
        for future in as_completed(futures):
            row = future.result(); rows.append(row); name = row['probe']['name']
            groups.setdefault(name, {})[row['candidate']] = row
            if len(groups[name]) == len(bank):
                ordered = [groups[name][candidate] for candidate, _ in bank]
                original = ordered[0]['result']; expected = old[name]['result']
                for key in ['accepted', 'completed_model_steps', 'failure', 'terminal', 'rank']:
                    if original[key] != expected[key]:
                        if (original.get('failure') or {}).get('reason') != 'declared_compute_budget':
                            raise ValueError('Original prediction changed: ' + name + '/' + key)
                complete = not any((r['result']['failure'] or {}).get('reason') == 'declared_compute_budget' for r in ordered)
                selected = select_prediction(ordered) if complete else None
                item = dict(probe=name, complete_candidate_comparison=complete,
                    candidate_feasible={r['candidate']: r['result']['accepted'] for r in ordered},
                    chosen_candidate=selected['candidate'] if selected else None,
                    status='model_feasible_plan' if selected else ('no_model_feasible_plan' if complete else 'compute_incomplete'),
                    baseline_accepted=expected['accepted'], native_execution=False)
                selections.append(item); print(json.dumps(item), flush=True); checkpoint()
    for path, digest in hashes.items():
        if sha(path) != digest: raise ValueError('Source changed during evaluation ' + path)
    selections.sort(key=lambda r: list(old).index(r['probe']))
    complete = len(selections) == 13 and all(r['complete_candidate_comparison'] for r in selections)
    result = dict(scope=__doc__, configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source_sha256=hashes, bank=[dict(name=name, knots_rad=knots) for name,knots in bank],
        prediction_count=len(rows), predictions=rows, selections=selections,
        all_thirteen_completed=complete, selected_model_feasible=sum(r['chosen_candidate'] is not None for r in selections),
        original_model_successes_retained=complete and all(r['chosen_candidate']=='baseline' for r in selections if r['baseline_accepted']),
        wall_seconds=time.monotonic()-started, native_execution=False, valid_expert_labels=False,
        limitations=['Selected outcomes are on the planning model, not independent native execution.',
            'All probes are previously inspected development states, not a success-rate estimate.',
            'Full-episode initial selection does not yet implement replanning at later control ticks.',
            'No candidate feasible or compute incomplete means no certified model proposal; it is not permission to execute the least-bad plan.'])
    args.output.write_text(json.dumps(plain(result), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
