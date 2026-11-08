"""Replan the failed finite model probes without widening native admission.

This tests whether current-state optimization can repair a fixed-plan failure.
It does not create native perturbation rollouts or certified expert labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_native_replanned_suffix_v1 import validate
from scripts.check_constrained_recovery_suffix_v1 import source_context, state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.current_state_recovery_service_v1 import CurrentStateRecoveryServiceV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--probes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds-per-case', type=float, default=600.)
    args = parser.parse_args()
    if args.output.exists() or not 120 <= args.max_seconds_per_case <= 600:
        raise ValueError('Fresh output and bounded search required')
    source = json.loads(args.source.read_text())
    probes = json.loads(args.probes.read_text())
    tick, _ = validate(source)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if (not probes['nominal_exact'] or probes['compute_expired_probes'] or
            probes['source_sha256'][str(args.source.resolve())] != sha(args.source)):
        raise ValueError('Completed probes must match the fixed actual-state suffix')
    failures = [r for r in probes['rows'] if not r['result']['accepted']]
    if not 1 <= len(failures) <= 2:
        raise ValueError('This bounded follow-up supports one or two recorded failures')
    _, knots, _, audit, initargs = source_context(source['configuration']['source'])
    initialize(*initargs)
    plan = dict(original_knots_rad=knots, corrections=[dict(control_tick=tick,
        previous_command=source['query']['target'], knots_rad=source['best']['corrections_rad'])])
    pins = dict(probes['source_sha256'])
    pins[str(args.probes.resolve())] = sha(args.probes)
    for name, module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            pins[str(path)] = sha(path)
    pins[str(Path(__file__).resolve())] = sha(__file__)
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Changed input: ' + path)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for probe in failures:
        state = state_from_row(probe['query'])
        service = CurrentStateRecoveryServiceV1(initargs, audit, pins, workers=4,
            max_seconds=args.max_seconds_per_case)
        print(json.dumps(dict(starting_probe=probe['probe']['name'])), flush=True)
        try:
            result = service.solve(state, tick, plan)
        finally:
            service.close()
        warm = result['records'][0]['row']['result']
        if (warm['accepted'] or warm['completed_model_steps'] != probe['result']['completed_model_steps'] or
                warm['failure'] != probe['result']['failure']):
            raise ValueError('Failed fixed-plan probe did not reproduce before search')
        for name in ['q', 'v', 'contacts', 'target']:
            if not np.array_equal(result['query'][name], probe['query'][name]):
                raise ValueError('Search changed supplied state/history: ' + name)
        rows.append(dict(probe=probe['probe'], result=result))
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(rows), allow_nan=False) + '\n')
        tmp.replace(args.output.with_suffix('.partial.json'))
        print(json.dumps(dict(probe=probe['probe']['name'], accepted=result['accepted'],
            wall_seconds=result['wall_seconds'], stages=[r['stage'] for r in result['records']])), flush=True)
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('Input changed during search: ' + path)
    result = dict(scope=__doc__, source_sha256=pins, rows=rows,
        repaired_model_probes=sum(r['result']['accepted'] for r in rows),
        native_execution=False, query_admission_widened=False, valid_expert_labels=False,
        limitations=['Only the two failed single-axis model perturbations are tested.',
            'Each case permits up to 600 seconds of CPU planning; no real-time claim.',
            'A new model solution still requires ground-start native execution and uncertainty checks.'])
    tmp = args.output.with_suffix('.tmp')
    tmp.write_text(json.dumps(plain(result), indent=2, allow_nan=False) + '\n')
    tmp.replace(args.output)


if __name__ == '__main__':
    main()
