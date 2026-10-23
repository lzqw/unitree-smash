#!/usr/bin/env python3
"""Compare original and extended same-tolerance sweep compute budgets.

This is a query diagnostic only, with no PhysX rollout or collision response.
CUDA invocation must use run_gpu_job.py like every other GPU calculation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.physics.native import swept_sphere_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'metrics.json').exists() or (args.output/'completed.json').exists():
        raise ValueError('Use a new output directory without previous diagnostic results')
    q = [1., 0., 0., 0.]
    def case(name, start, end, expected, center_start=None, center_end=None, quat_end=None):
        return dict(name=name, start=start, end=end, center_start=center_start or [0., 0., 0.],
                    center_end=center_end or [0., 0., 0.], quat_start=q, quat_end=quat_end or q,
                    half_extents=[.0075, .11, .11], radius=.025, expected_hit=expected)
    cases = [case('analytic_face_hit', [-.2,0,0], [.2,0,0], True),
             case('analytic_clear_miss', [-.2,.14,0], [.2,.14,0], False),
             case('analytic_tangent_hit', [-.2,.135,0], [.2,.135,0], True),
             case('analytic_translating_hit', [0,0,0], [0,0,0], True, [-.2,0,0], [.2,0,0]),
             case('analytic_rotating_hit', [.08,.08,0], [.08,.08,0], True,
                  quat_end=[math.sqrt(.5),0,0,-math.sqrt(.5)]),
             case('analytic_double_cover_hit', [-.2,0,0], [.2,0,0], True, quat_end=[-1.,0,0,0])]
    for run in ['frozen_candidate_matrix_fine768_01', 'body_feedback_matrix_fine1280_01', 'forward_body_matrix_fine1280_01']:
        source = ROOT/'reports/G07'/run/'unresolved_sweep_inputs.json'
        for index, row in enumerate(json.loads(source.read_text())):
            cases.append(dict(row, name=f'{run}:{index}', source=str(source.relative_to(ROOT)),
                              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    results = []
    for row in cases:
        entry = dict(name=row['name'], source=row.get('source'), source_sha256=row.get('source_sha256'), results={})
        for device, dtype, iterations, label in [(args.device, torch.float32,256,'device_f32_256'),
                                                (args.device,torch.float32,4096,'device_f32_4096'),
                                                ('cpu',torch.float64,4096,'cpu_f64_4096')]:
            values = [torch.tensor([row[k]], dtype=dtype, device=device) for k in
                      ['start','end','center_start','center_end','quat_start','quat_end']]
            hit, fraction, unknown = swept_sphere_box(*values,row['half_extents'],row['radius'],iterations=iterations,tolerance=1e-6)
            entry['results'][label] = dict(hit=bool(hit[0]), unresolved=bool(unknown[0]), fraction=float(fraction[0]))
        fine, reference = entry['results']['device_f32_4096'], entry['results']['cpu_f64_4096']
        entry['passed'] = (not fine['unresolved'] and not reference['unresolved'] and fine['hit']==reference['hit']
                           and ('expected_hit' not in row or fine['hit']==row['expected_hit']))
        if row['name']=='analytic_face_hit':
            entry['passed'] &= abs(fine['fraction']-.41875)<1e-5
        results.append(entry)
    report = dict(kind='Same-query CPU/CUDA diagnostic only; not a physical rollout or policy acceptance',
                  device=args.device, tolerance_m=1e-6, original_iterations=256, extended_iterations=4096,
                  all_original_failures_retained=True, collision_response_changed=False,
                  cases=results, passed=all(r['passed'] for r in results))
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(device=args.device,passed=report['passed'],cases=len(results),
                         legacy_unresolved=sum(r['results']['device_f32_256']['unresolved'] for r in results),
                         extended_unresolved=sum(r['results']['device_f32_4096']['unresolved'] for r in results))))
    if not report['passed']:
        raise RuntimeError('Extended query budget still unresolved or differs from analytic/higher precision checks')
    (args.output/'completed.json').write_text(json.dumps(dict(query_cases=len(results)))+'\n')


if __name__ == '__main__':
    main()
