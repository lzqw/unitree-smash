"""Instrument the repeated timed recovery, without changing any commands."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['parent-plan', 'output', 'native-output']:
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.native_output.exists():
        raise ValueError('Fresh output required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent = json.loads(a.parent_plan.read_text())
    if json.loads((Path(parent['output'])/'job.json').read_text())['status'] != 'SUCCEEDED':
        raise ValueError('Completed parent required')
    pins = dict(parent['source_sha256'])
    for name, digest in pins.items():
        if sha(name) != digest:
            raise ValueError('Changed source '+name)
    command = copy.deepcopy(parent['command'])
    old = next(Path(v) for v in command if v.endswith('evaluate_timed_recovery.py'))
    if sha(old) != parent['entry_sha256']:
        raise ValueError('Parent entry changed')
    source = old.read_text()
    changes = {
        '        head_observer=NativeHeadGroundObservationV1(env,args.output)':
        '        head_observer=NativeHeadGroundObservationV1(env,args.output)\n'
        '        from scripts.native_timed_impact_observer_v1 import NativeTimedImpactObserverV1\n'
        '        timed_impact_observer=NativeTimedImpactObserverV1(env,args.output)',
        "    finally:\n        if 'head_observer' in locals():":
        "    finally:\n        if 'timed_impact_observer' in locals():\n"
        "            timed_impact_observer.save()\n            timed_impact_observer.detach()\n"
        "        if 'head_observer' in locals():",
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError('Unexpected evaluator lifecycle')
        source = source.replace(before, after)
    # Observation wrappers must be detached in reverse installation order.
    installed = [source.index(x) for x in [
        'isolation_observer=NativeIsolationMicrotraceV1(',
        'head_observer=NativeHeadGroundObservationV1(',
        'timed_impact_observer=NativeTimedImpactObserverV1(']]
    removed = [source.index(x) for x in [
        'timed_impact_observer.detach()', 'head_observer.detach()',
        'isolation_observer.detach()']]
    if installed != sorted(installed) or removed != sorted(removed):
        raise ValueError('Observer stack must unwind in reverse order')
    ast.parse(source)
    a.output.mkdir(parents=True)
    entry = a.output/'evaluate_timed_impact.py'; entry.write_text(source)
    command = [str(entry.resolve()) if v == str(old) else v for v in command]
    for i, v in enumerate(command[:-1]):
        if v == '--output':
            command[i+1] = str(a.native_output.resolve())
    for path in [a.parent_plan, Path(__file__), entry,
                 ROOT/'scripts/native_timed_impact_observer_v1.py',
                 ROOT/'scripts/native_foot_impact_observer_v1.py']:
        pins[str(path.resolve())] = sha(path)
    plan = dict(parent, output=str(a.native_output.resolve()), command=command,
                entry_sha256=sha(entry), source_sha256=pins,
                observation_parent=parent['output'], observer_controls_changed=False,
                post_command=['bash','scripts/python.sh','scripts/audit_native_experimental_recovery_v1.py',
                    '--run',str(a.native_output.resolve()),'--bundle',parent['bundle'],
                    '--output',str((a.output/'native_audit.json').resolve())])
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'plan':str(a.output/'plan.json'), 'frames':[288*160,300*160],
                      'new_controller':False,'native_execution':False}))


if __name__ == '__main__':
    main()
