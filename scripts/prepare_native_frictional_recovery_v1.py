"""Prepare a pinned native entry without changing an already running evaluator."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = 'c758bf1327fdfe8d009f55c9ba0fe11b3e3834fdc213dd78e46e505600eccd63'


def prepare(source, output):
    source = Path(source); output = Path(output)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(source) != SOURCE_SHA256 or output.exists():
        raise ValueError('Require unchanged source and a fresh destination')
    text = source.read_text()
    replacements = [
        ("parser.add_argument('--recovery-sequence-case', type=int, required=True)",
         "parser.add_argument('--recovery-sequence-case', type=int, required=True)\n"
         "parser.add_argument('--frictional-warm-plan', type=Path, required=True)\n"
         "parser.add_argument('--frictional-sphere-audit', type=Path, required=True)"),
        ('recovery_sequence_bundle=str(args.recovery_sequence_bundle.resolve()),',
         'recovery_sequence_bundle=str(args.recovery_sequence_bundle.resolve()),\n'
         '        frictional_warm_plan=str(args.frictional_warm_plan.resolve()),\n'
         '        frictional_warm_plan_sha256=digest(args.frictional_warm_plan),\n'
         '        frictional_sphere_audit=str(args.frictional_sphere_audit.resolve()),\n'
         '        frictional_sphere_audit_sha256=digest(args.frictional_sphere_audit),'),
        ('from scripts.native_receding_recovery_v1 import NativeRecedingRecoveryV1 as NativeRecoverySequenceReplay',
         'from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1 as NativeRecoverySequenceReplay'),
        ('args.recovery_sequence_case,limits,digest(args.checkpoint),args.output)',
         'args.recovery_sequence_case,limits,digest(args.checkpoint),args.output,\n'
         '            args.frictional_warm_plan,args.frictional_sphere_audit)'),
        ('        env._diagnostic_observer=None\n        env.close()',
         "        if 'native_replay' in locals():\n            native_replay.planner.close()\n"
         '        env._diagnostic_observer=None\n        env.close()'),
    ]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError('Changed native entry contract: '+before)
        text = text.replace(before, after)
    ast.parse(text)
    output.mkdir(parents=True)
    entry = output/'evaluate_frictional.py'
    entry.write_text(text)
    record = dict(source=str(source.resolve()), source_sha256=digest(source),
        entry=str(entry.resolve()), entry_sha256=digest(entry),
        preparer_sha256=digest(__file__), native_execution=False,
        teacher_model='frictional_spheres_v1', horizon_steps=80)
    (output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source,args.output)))
