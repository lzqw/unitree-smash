#!/usr/bin/env python3
"""Separate actual applied-command errors from fallback and contact-flag changes.

These are adjacent 10 ms native states. Contact flags are force observations,
not an independent geometric contact detector. Different runs reach different
states; their errors do not constitute a matched controller ablation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                maximum=float(a.max())) if len(a) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing evidence')
    runs = []
    for path in a.audit:
        data = json.loads(path.read_text())
        for f, digest in data['source_sha256'].items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest() != digest:
                raise ValueError('Prediction source changed: ' + f)
        groups = {}
        for name, select in [
                ('all', lambda r: True),
                ('applied', lambda r: r['applied']),
                ('applied_force_flag_unchanged', lambda r: r['applied'] and not r['observed_contact_change']),
                ('applied_force_flag_changed', lambda r: r['applied'] and r['observed_contact_change'])]:
            rows = [r for r in data['records'] if select(r)]
            models = {}
            for model in data['summary']:
                valid = [r['models'][model] for r in rows if r['models'][model]['ok']]
                metrics = ['leg_position_error_max_rad', 'arm_position_error_max_rad',
                    'base_linear_velocity_error_mps', 'base_angular_velocity_error_rad_s']
                violations = {}
                for key in ['normal_force_violation_n', 'friction_violation_n', 'joint_velocity_violation_rad_s']:
                    values = [r['physical_model_diagnostics'][key] for r in valid if r['physical_model_diagnostics'] is not None]
                    violations[key] = dict(stats=stats(values), above_1e_minus3=sum(v > 1e-3 for v in values))
                models[model] = dict(failed=len(rows)-len(valid),
                    errors={k: stats([r[k] for r in valid]) for k in metrics}, diagnostic_violations=violations)
            groups[name] = dict(samples=len(rows), models=models)
        runs.append(dict(run=data['source'], samples=data['samples'], groups=groups,
                         audit=str(path.resolve()), audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(scope=__doc__, native_new_execution=False, independent_geometry_contact=False,
                  matched_between_run_ablation=False, runs=runs,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), samples=[r['samples'] for r in runs])))


if __name__ == '__main__':
    main()
