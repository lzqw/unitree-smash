"""Prepare read-only physics-step observation using the verified passive bank entry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Fresh preparation required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan_path = root/'reports/landing_native_query_bank_followup_20261108/plan.json'
    plan = json.loads(plan_path.read_text())
    audit_path = plan_path.parent/'query_bank_audit.json'
    if not json.loads(audit_path.read_text())['passed']:
        raise ValueError('Verified passive source required')
    source = Path(plan['command'][plan['command'].index('-u')+1])
    if sha(source) != plan['entry_sha256']:
        raise ValueError('Executed entry changed')
    text = source.read_text()
    changes = {
        '        query_bank = NativeRecoveryQueryBankV1(env, native_replay.model, args.output)':
        '        query_bank = NativeRecoveryQueryBankV1(env, native_replay.model, args.output)\n'
        '        from scripts.native_foot_impact_observer_v1 import NativeFootImpactObserverV1\n'
        '        impact_observer = NativeFootImpactObserverV1(env, args.output)',
        "    finally:\n        if 'query_bank' in locals():":
        "    finally:\n        if 'impact_observer' in locals():\n"
        "            impact_observer.save()\n            impact_observer.detach()\n"
        "        if 'query_bank' in locals():",
    }
    for before, after in changes.items():
        if text.count(before) != 1:
            raise ValueError('Unexpected observer lifecycle: '+before)
        text = text.replace(before, after)
    ast.parse(text)
    output.mkdir(parents=True)
    entry = output/'evaluate_foot_impact.py'
    entry.write_text(text)
    files = [plan_path,audit_path,source,Path(__file__).resolve(),
             root/'scripts/native_foot_impact_observer_v1.py']
    result = dict(entry=str(entry),entry_sha256=sha(entry),
        source_sha256={str(p):sha(p) for p in files},paired_run=plan['output'],
        control_changes=False,physical_changes=False,scoring_changes=False,
        lanes=[85,87],physics_frames=[297*160,306*160],native_execution=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.output)))
