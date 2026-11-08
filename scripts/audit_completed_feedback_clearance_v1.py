"""Measure authored surface margins on complete model-feedback trajectories.

Read-only 10 ms endpoint audit, including the actual initial state. Neither
positive sampled clearance nor model stability certifies native execution.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.event_recovery_manager_v1 import authored_clearance_state
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'run', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--feedback-file', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh audit output required')
    read = lambda path: json.loads(path.read_text())
    cfg = read(a.configuration)
    nominal = read(a.run / 'native_command_inputs.json')['default_joint_position_rad']
    c = PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']), nominal,
                                         cfg['model_reference_library'])
    pins = {}
    records = []
    for path in a.feedback_file:
        data = read(path)
        inputs_path = path.parent / 'inputs.json'
        for filename, digest in read(inputs_path)['source_sha256'].items():
            if sha(filename) != digest:
                raise ValueError('Changed feedback source ' + filename)
            if filename in pins and pins[filename] != digest:
                raise ValueError('Conflicting sources')
            pins[filename] = digest
        trace = data['trajectory']
        initial = data['initial_actual_state']
        if (data['failure'] is not None or not data['full_model_path'] or data['native_success']
                or data['valid_expert_labels'] or len(trace) != 800 - initial['tick']
                or any(row['tick'] != initial['tick'] + i for i, row in enumerate(trace))):
            raise ValueError('Complete declared model feedback required')
        points = []
        states = [(initial['tick'], initial)] + [(row['tick'] + 1, dict(row, previous=row['target'])) for row in trace]
        for tick, row in states:
            state = state_from_record(c.model, row)
            surfaces = authored_clearance_state(c.geometry, state)
            head = min((dict(body=name, **surfaces[name]) for name in ['Head_lower', 'Head_upper']),
                       key=lambda value: value['gap_m'])
            # The shared geometry also includes labels such as Racket::net.
            # Their distance is to an obstacle, not to the ground plane.
            nonfoot = min((dict(body=name, **value) for name, value in surfaces.items()
                           if not name.endswith('_foot') and '::' not in name),
                          key=lambda value: value['gap_m'])
            obstacles = [dict(body=name, **value) for name, value in surfaces.items() if '::' in name]
            points.append(dict(tick=tick, time_s=tick * .01, head=head, nonfoot=nonfoot,
                nearest_obstacle=None if not obstacles else min(obstacles, key=lambda value: value['gap_m'])))
        records.append(dict(state_key=data['state_key'], samples=len(points),
            minimum_head=min(points, key=lambda row: row['head']['gap_m']),
            minimum_nonfoot=min(points, key=lambda row: row['nonfoot']['gap_m']),
            initial=points[0], final=points[-1],
            model_stable_s=data['final_model_stable_s'],
            command_sources=dict(Counter(row['source'] for row in trace)),
            source_feedback=str(path.resolve())))
        pins.update({str(f.resolve()): sha(f) for f in [path, inputs_path]})
    files = [Path(__file__), a.configuration, a.run / 'native_command_inputs.json', Path(cfg['model_reference_library'])]
    files += [Path(m.__file__) for name, m in list(sys.modules.items())
              if name.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(f.resolve()): sha(f) for f in files})
    result = dict(records=records, source_sha256=pins, sample_period_s=.01,
        native_execution=False, native_success=False, valid_expert_labels=False,
        ground_metric_excludes_paired_obstacle_labels=True,
        limitation='Authored collision surface at control boundaries, not all physics substeps or cooked PhysX contact geometry.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write(a.output, result)
    print(json.dumps(dict(records=records, native_success=False)))


if __name__ == '__main__':
    main()
