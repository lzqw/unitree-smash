#!/usr/bin/env python3
"""Isolated CPU comparison; never patches files or a running native process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', choices=['original', 'batched'])
    a = p.parse_args()
    if a.worker:
        if a.worker == 'batched':
            # Only this fresh child interpreter uses the alternative builder.
            import scripts.interval_command_support_qp_v1 as module
            from scripts.batched_interval_constraints_v1 import interval_constraints
            module.interval_constraints = interval_constraints
        from scripts.audit_minimum_command_mpc_v1 import main as audit
        sys.argv = [sys.argv[0], '--run', str(a.run), '--output', str(a.output)]
        audit()
        return
    a.output.mkdir(parents=True, exist_ok=False)
    elapsed = {}
    for variant in ('original', 'batched'):
        started = time.perf_counter()
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with (a.output/f'{variant}.log').open('x') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                '--run', str(a.run.resolve()), '--output', str((a.output/f'{variant}.json').resolve()),
                '--worker', variant], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        elapsed[variant] = time.perf_counter()-started
    import numpy as np
    source = {k: json.loads((a.output/f'{k}.json').read_text()) for k in elapsed}
    old, new = source['original']['records'], source['batched']['records']
    if [(r['case'], r['tick']) for r in old] != [(r['case'], r['tick']) for r in new]:
        raise ValueError('Replay sampling changed')
    differences = []; maximum_target_error = 0.
    for x, y in zip(old, new):
        for model in x['models']:
            left, right = x['models'][model], y['models'][model]
            for key in ('first_accepted', 'horizon_accepted', 'failed_stage', 'stage_count', 'rejection'):
                if left[key] != right[key]:
                    differences.append(dict(case=x['case'], tick=x['tick'], model=model, field=key,
                                            original=left[key], batched=right[key]))
            if 'target' in left and 'target' in right:
                maximum_target_error = max(maximum_target_error,
                    float(np.max(abs(np.asarray(left['target'])-right['target']))))
    files = [Path(__file__), ROOT/'scripts/batched_interval_constraints_v1.py',
             ROOT/'tests/test_batched_interval_constraints_v1.py']
    files += [a.output/f'{k}.json' for k in elapsed]
    result = dict(scope=__doc__, native_new_execution=False, installed_in_native_controller=False,
        samples=len(old), full_plan_decision_differences=differences,
        first_command_max_difference_rad=maximum_target_error,
        command_equivalence_tolerance_rad=1e-10,
        passed=not differences and maximum_target_error <= 1e-10,
        elapsed_seconds=elapsed,
        total_cpu_replay_speedup=elapsed['original']/elapsed['batched'],
        summaries={k:v['summary'] for k,v in source.items()},
        limitations=['One sequential timing pair; scheduling and cache effects remain.',
                     'Matches sampled plans and first commands, not native closed-loop trajectories.',
                     'No physical constraints, objective or horizon shortened.'],
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if not result['passed']:
        raise ValueError('Batch implementation changed replay decisions or commands')


if __name__ == '__main__':
    main()
