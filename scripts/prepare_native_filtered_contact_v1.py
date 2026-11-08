"""Prepare a separate passive ground/pair measurement of the timed parent."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['parent-plan', 'replay-parent', 'output', 'native-output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.native_output.exists():
        raise ValueError('Fresh output required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent = json.loads(a.parent_plan.read_text())
    for path in [Path(parent['output']), a.replay_parent]:
        if json.loads((path / 'job.json').read_text())['status'] != 'SUCCEEDED':
            raise ValueError('Completed physical parents required')
    pins = dict(parent['source_sha256'])
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Changed parent source ' + path)
    command = copy.deepcopy(parent['command'])
    old = next(Path(v) for v in command if v.endswith('evaluate_timed_recovery.py'))
    if sha(old) != parent['entry_sha256']:
        raise ValueError('Different parent entry')
    source = old.read_text()
    changes = {
        '        head_observer=NativeHeadGroundObservationV1(env,args.output)':
        '        head_observer=NativeHeadGroundObservationV1(env,args.output)\n'
        '        from scripts.native_ground_pair_observer_v1 import NativeGroundPairObserverV1\n'
        '        filtered_observer=NativeGroundPairObserverV1(env,args.output)',
        "    finally:\n        if 'head_observer' in locals():":
        "    finally:\n        if 'filtered_observer' in locals():\n"
        "            filtered_observer.save()\n            filtered_observer.detach()\n"
        "        if 'head_observer' in locals():",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError('Unexpected wrapper lifecycle')
        source = source.replace(before, after)
    installation = [source.index(x) for x in ['isolation_observer=NativeIsolationMicrotraceV1(',
        'head_observer=NativeHeadGroundObservationV1(', 'filtered_observer=NativeGroundPairObserverV1(']]
    removal = [source.index(x) for x in ['filtered_observer.detach()', 'head_observer.detach()', 'isolation_observer.detach()']]
    if installation != sorted(installation) or removal != sorted(removal):
        raise ValueError('Observer cleanup must reverse installation')
    ast.parse(source)
    a.output.mkdir(parents=True)
    entry = a.output / 'evaluate_filtered_contact.py'; entry.write_text(source)
    command = [str(entry.resolve()) if v == str(old) else v for v in command]
    for i, v in enumerate(command[:-1]):
        if v == '--output':
            command[i + 1] = str(a.native_output.resolve())
    paths = [a.parent_plan, Path(__file__), entry,
        ROOT / 'scripts/native_ground_pair_observer_v1.py', ROOT / 'scripts/native_timed_impact_observer_v1.py',
        ROOT / 'scripts/native_foot_impact_observer_v1.py', ROOT / 'scripts/audit_filtered_contact_native_v1.py',
        ROOT / 'scripts/audit_timed_impact_observation_v2.py',
        Path('/home/autodrive/Isaac/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py')]
    tensor_root = Path('/home/autodrive/Isaac/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache')
    api = list(tensor_root.glob('omni.physics.tensors-*/omni/physics/tensors/impl/api.py'))
    if len(api) != 1:
        raise ValueError('One pinned installed tensor API required')
    paths += api
    for path in paths:
        pins[str(path.resolve())] = sha(path)
    plan = dict(parent, output=str(a.native_output.resolve()), command=command,
        entry_sha256=sha(entry), source_sha256=pins, observation_parent=str(a.replay_parent.resolve()),
        observer_controls_changed=False, post_command=['bash', 'scripts/python.sh',
            'scripts/audit_filtered_contact_native_v1.py', '--run', str(a.native_output.resolve()),
            '--parent', str(a.replay_parent.resolve()), '--bundle', parent['bundle'],
            '--output', str((a.output / 'native_audit.json').resolve())])
    (a.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan=str(a.output / 'plan.json'), native_execution=False, controls_changed=False)))


if __name__ == '__main__':
    main()
