"""Check declared native state-transition variant, not physical success.

Only the native controller import/config label changes. Verify all recorded
causal entry commands against V15; later recovery actions may intentionally
change. No claim to preserve old successful complete native trajectories.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_budgeted_settling_feedback_v1 import sha, write
from scripts.pure_control_state_settling_v23 import PureControlStateSettlingV23
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record

OLD = 'from scripts.native_pure_control_stance_foot_reference_v15 import NativePureControlStanceFootReferenceV15 as NativePureControlRecoveryV1'
NEW = 'from scripts.native_pure_control_state_settling_v23 import NativePureControlStateSettlingV23 as NativePureControlRecoveryV1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['parent-plan', 'entry', 'configuration', 'feedback', 'fixture-run', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    if a.output.exists():
        raise ValueError('Fresh check receipt required')
    parent = read(a.parent_plan)
    command = parent['command']
    old_entry = next(Path(x) for x in command if x.endswith('/evaluate_pure_control.py'))
    original = old_entry.read_text()
    text = a.entry.read_text()
    if original.count(OLD) != 1 or text != original.replace(OLD, NEW):
        raise ValueError('Only declared controller import may change')
    ast.parse(text)
    call = 'action=native_replay.action(tick,obs,action,finished)'
    if text.count(call) != 1:
        raise ValueError('Missing unique final command owner')
    block = text[text.index(call):text.index('observation, _, done, _ = wrapped.step(action)')]
    if 'action = ' in block or 'action=' in block.split('\n', 1)[1]:
        raise ValueError('Second final action writer')
    old_cfg = read(command[command.index('--pure-control-configuration') + 1])
    cfg = read(a.configuration)
    for key in set(cfg) | set(old_cfg):
        if key not in ['method', 'controller'] and cfg.get(key) != old_cfg.get(key):
            raise ValueError('Unexpected execution configuration change: ' + key)
    if cfg['controller'] != 'pure_control_state_settling_v23':
        raise ValueError('Wrong controller identity')
    inputs = read(a.feedback / 'inputs.json')
    feedback = read(a.feedback / 'result.json')
    if feedback['state'] != 'COMPLETED_COARSE_SETTLING_COUNTERFACTUAL' or inputs['physical_substeps'] != 160:
        raise ValueError('Complete original-physics feedback evidence required')
    pins = {}
    for data in [parent, inputs]:
        for path, digest in data['source_sha256'].items():
            if sha(path) != digest or (path in pins and pins[path] != digest):
                raise ValueError('Changed/conflicting evidence: ' + path)
            pins[path] = digest
    if read(a.fixture_run / 'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed native causal entry evidence required')
    for name in ['pure_control_commands.npz', 'native_command_inputs.json']:
        path = a.fixture_run / name
        if inputs['source_sha256'].get(str(path.resolve())) != sha(path):
            raise ValueError('Feedback and entry checks use different native states')
    for row in feedback['results']:
        if (row['mode'] != 'state_settling' or row['failure'] is not None
                or row['steps'] != row['required_steps'] or row['final_model_stable_s'] < .8):
            raise ValueError('Feedback did not complete the original final stability window')
    nominal = read(a.fixture_run / 'native_command_inputs.json')['default_joint_position_rad']
    with np.load(a.fixture_run / 'pure_control_commands.npz') as z:
        data = {k: z[k].copy() for k in ['tick', 'lane', 'time_s', 'q', 'v', 'previous', 'ground_force']}
    rows = [json.loads(line) for line in (a.fixture_run / 'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(rows) != len(data['tick']):
        raise ValueError('Misaligned native records')
    checks = []
    bank = read(cfg['model_reference_library'])
    for result in feedback['results']:
        key = result['state_key']
        evidence_path = a.feedback / (key.replace(':', '_') + '.json')
        evidence = read(evidence_path)
        record = evidence['initial_actual_state']
        candidates = [r for r in bank['references'] if r['source_case'] == record['case']
            and r.get('source_actual_tick') == record['tick']]
        if len(candidates) != 1:
            raise ValueError('Missing intended new reference')
        planned = read(candidates[0]['source_search'])
        if planned['initial_actual_state'] != record:
            raise ValueError('Reference and feedback use different initial states')
        snapshots = [Path(f) for f in inputs['source_sha256'] if Path(f).name == 'state_' + key.replace(':', '_') + '.json'
            and Path(f).parent.name == 'candidate_snapshots']
        if not snapshots or not all(read(f)['fine']['trajectory'] == planned['fine']['trajectory'] for f in snapshots):
            raise ValueError('Feedback was checked against another reference trajectory')
        pins[str(evidence_path.resolve())] = sha(evidence_path)
    for lane in sorted(set(data['lane'].tolist())):
        i = int(np.flatnonzero(data['lane'] == lane)[0])
        row = rows[i]
        if (row['tick'], row['lane']) != (int(data['tick'][i]), lane) or not row['phase']['owns_final_recovery_command']:
            raise ValueError('Not a causal release observation')
        controllers = [cls(PositionPDInterface(**cfg['interface']), nominal, cfg['model_reference_library'])
            for cls in [PureControlStanceFootReferenceV15, PureControlStateSettlingV23]]
        record = {k: data[k][i] for k in ['q', 'v', 'previous', 'ground_force']}
        record['phase'] = row['phase']
        targets = [c.command(state_from_record(c.model, record), row['phase']['phase'], float(data['time_s'][i]))[0]
            for c in controllers]
        error = float(np.max(abs(targets[0] - targets[1])))
        if error > 2e-6:
            raise ValueError('First recovery command unexpectedly changed')
        checks.append(dict(case=int(lane), tick=row['tick'], maximum_entry_error_rad=error))
    if not checks or len(feedback['results']) != 2:
        raise ValueError('Incomplete declared evidence coverage')
    files = [Path(__file__), a.parent_plan, a.entry, a.configuration, a.feedback / 'inputs.json',
        a.feedback / 'result.json', a.fixture_run / 'pure_control_diagnostics.jsonl',
        ROOT / 'scripts/native_pure_control_state_settling_v23.py']
    files += [Path(m.__file__) for n, m in list(sys.modules.items())
        if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins.update({str(f.resolve()): sha(f) for f in files})
    receipt = dict(passed=True, source_sha256=pins, causal_entry_checks=checks,
        checked_model_paths=feedback['results'], native_execution=False,
        native_success=False, valid_expert_labels=False, hardware_unchanged=True,
        single_final_writer=True, post_entry_actions_intentionally_may_change=True,
        old_native_success_retention_unproven=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write(a.output, receipt)
    print(json.dumps(dict(passed=True, causal_entries=len(checks), model_paths=len(feedback['results']))))


if __name__ == '__main__':
    main()
