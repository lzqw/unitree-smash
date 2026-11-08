"""Report authored head/body clearance and approach speed along model feedback.

Uses 10 ms model states, not cooked PhysX contacts or substep minima. Small
clearance is a diagnostic margin, not a new success threshold or certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = read(args.feedback/'result.json')
    if args.output.exists() or result['state'] != 'COMPLETED_FIRST_MATCH_FEEDBACK_MODEL':
        raise ValueError('Fresh audit of completed first-match feedback required')
    for filename,digest in read(args.feedback/'inputs.json')['source_sha256'].items():
        if sha(filename) != digest:
            raise ValueError('Changed model audit source '+filename)
    geometry_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    model = LoadedPinocchioModelV1()
    geometry = AuthoredGroundClearance(geometry_path, model)
    bodies = [shape['body'] for shape in geometry.shapes]
    head = np.array([i for i,b in enumerate(bodies) if b in ['Head_lower', 'Head_upper']])
    if len(head) == 0:
        raise ValueError('Missing actual head collision geometry')
    summaries, files = [], [Path(__file__), args.feedback/'inputs.json', args.feedback/'result.json', geometry_path]
    for item in result['results']:
        source = args.feedback/('case_%03d.json'%item['case'])
        data = read(source)
        if not data['model_terminal_window_passed']:
            continue
        files.append(source)
        initial = data['initial_actual_state']
        observations = [(initial['tick'], initial['q'], initial['v'])]
        observations += [(r['tick']+1, r['q'], r['v']) for r in data['trajectory']]
        points = []
        for tick,q,v in observations:
            gaps,jacobian = geometry.evaluate(np.asarray(q), jacobian=True)
            rates = jacobian@np.asarray(v)
            lower = int(np.argmin(gaps))
            lower_head = int(head[np.argmin(gaps[head])])
            def surface(index):
                return dict(body=bodies[index], gap_m=float(gaps[index]),
                    vertical_surface_rate_m_s=float(rates[index]))
            points.append(dict(tick=tick, time_s=tick*.01,
                nonfoot=surface(lower), head=surface(lower_head)))
        summaries.append(dict(case=item['case'], samples=len(points),
            minimum_head=min(points, key=lambda r:r['head']['gap_m']),
            minimum_nonfoot=min(points, key=lambda r:r['nonfoot']['gap_m']),
            head_samples_below_01m=sum(r['head']['gap_m'] < .01 for r in points),
            nonfoot_samples_below_005m=sum(r['nonfoot']['gap_m'] < .005 for r in points),
            initial=points[0], final=points[-1]))
    files += [Path(module.__file__) for name,module in list(sys.modules.items())
        if name.startswith('scripts.') and getattr(module, '__file__', None)]
    output = dict(results=summaries, source_sha256={str(p.resolve()):sha(p) for p in files},
        sample_period_s=.01, contact_and_acceptance_thresholds_changed=False,
        native_contact_authority=False, native_success=False,
        limitation='Authored geometry at control-boundary model samples; no cooked collision or substep minimum claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
