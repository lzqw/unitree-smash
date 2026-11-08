"""Cross-test newly optimized recovery plans on the same finite state panel.

No reoptimization occurs during these probes. Original query history, physical
constraints, full-remainder horizon and integration resolution are unchanged.
This measures finite model coverage, not native robustness or teacher quality.
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
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.current_state_recovery_planner_v1 import predict_plan
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['source', 'probes', 'repairs', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1200.)
    args = parser.parse_args()
    if args.output.exists() or not 600 <= args.max_seconds <= 1800:
        raise ValueError('Fresh bounded model cross-test required')
    source = json.loads(args.source.read_text())
    probes = json.loads(args.probes.read_text())
    repairs = json.loads(args.repairs.read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if (not probes['nominal_exact'] or probes['compute_expired_probes'] or
            repairs['source_sha256'][str(args.probes.resolve())] != sha(args.probes) or
            probes['source_sha256'][str(args.source.resolve())] != sha(args.source)):
        raise ValueError('Inconsistent preceding model evidence')
    if probes['probe_count'] != 13 or len({r['probe']['name'] for r in probes['rows']}) != 13:
        raise ValueError('Original thirteen-state panel required')
    accepted = [r for r in repairs['rows'] if r['result']['accepted']]
    if not 1 <= len(accepted) <= 2 or any(not r['result']['independent_replay_exact'] for r in accepted):
        raise ValueError('One or two independently verified model repairs required')
    _, _, _, audit, context = source_context(source['configuration']['source'])
    pins = dict(repairs['source_sha256'])
    for path in [args.source, args.probes, args.repairs, Path(__file__)]:
        pins[str(path.resolve())] = sha(path)
    for name, module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve(); pins[str(path)] = sha(path)
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Changed model input: ' + path)
    began = time.monotonic(); deadline = began + args.max_seconds
    rows = []; nominal_predictions = {}; anchors = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=context) as pool:
        tasks = []
        for repair in accepted:
            label = repair['probe']['name']; result = repair['result']
            panel = sorted(probes['rows'], key=lambda r: (r['probe']['name'] != label,
                r['probe']['name'] != 'nominal', r['probe']['name']))
            for probe in panel:
                if probe['probe']['name'] == label and any(not np.array_equal(probe['query'][k], result['query'][k])
                        for k in ['q', 'v', 'contacts', 'target']):
                    raise ValueError('Repair query does not match the original failed probe')
                tasks.append((repair, probe))
        jobs = {pool.submit(predict_plan, repair['result']['plan'], probe['query'],
            probes['control_tick'], 0, audit, 160, deadline): (repair, probe) for repair, probe in tasks}
        for future in as_completed(jobs):
            repair, probe = jobs[future]; result = future.result()['result']
            label = repair['probe']['name']; name = probe['probe']['name']
            if name == label:
                old = repair['result']['prediction']['trajectory']; new = result['trajectory']
                exact = result['accepted'] and len(old) == len(new) and all(
                    np.array_equal(x[k], y[k]) for x, y in zip(old, new) for k in ['q', 'v', 'contacts', 'target'])
                anchors[label] = bool(exact)
            if name == 'nominal':
                nominal_predictions[label] = dict(plan=repair['result']['plan'], query=probe['query'], result=result)
            row = dict(plan=label, probe=probe['probe'], fixed_plan_accepted=result['accepted'],
                initial_state_and_history_preserved=True, completed_model_steps=result['completed_model_steps'],
                failure=result['failure'], terminal=result['terminal'], terminal_violations=result['terminal_violations'],
                trailing_model_window_s=result['trailing_model_window_s'])
            rows.append(row)
            print(json.dumps(dict(plan=label, probe=name, accepted=result['accepted'],
                completed_steps=result['completed_model_steps'], failure=result['failure'])), flush=True)
            tmp = args.output.with_suffix('.tmp'); tmp.write_text(json.dumps(plain(rows), allow_nan=False) + '\n')
            tmp.replace(args.output.with_suffix('.partial.json'))
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Input changed during model cross-test: ' + path)
    summary = [dict(plan=label, accepted=sum(r['fixed_plan_accepted'] for r in rows if r['plan'] == label),
        total=sum(r['plan'] == label for r in rows), original_repair_exact=anchors.get(label, False),
        nominal_accepted=nominal_predictions[label]['result']['accepted']) for label in nominal_predictions]
    output = dict(scope=__doc__, source_sha256=pins, rows=rows, summary=summary,
        nominal_predictions=nominal_predictions, wall_seconds=time.monotonic() - began,
        native_execution=False, query_admission_widened=False, valid_expert_labels=False,
        limitations=['Finite single-axis model checks are not a recoverable-domain certificate.',
            'Each plan is fixed throughout its thirteen probes; no failed probe is silently repaired.',
            'Native execution and combined/unseen perturbations remain untested.'])
    args.output.write_text(json.dumps(plain(output), indent=2, allow_nan=False) + '\n')
    if not all(anchors.values()) or len(anchors) != len(accepted):
        raise ValueError('At least one original repair could not be reproduced')


if __name__ == '__main__':
    main()
