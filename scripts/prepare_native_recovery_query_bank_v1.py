"""Prepare passive native recovery-state capture from an executed shadow entry."""
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
    source_plan = root / 'reports/landing_native_isolation_shadow_followup_20261108/plan.json'
    plan = json.loads(source_plan.read_text())
    source = Path(plan['command'][plan['command'].index('-u') + 1])
    if sha(source) != plan['entry_sha256']:
        raise ValueError('Executed source changed')
    text = source.read_text()
    substitutions = {
        '        finished = torch.zeros(count, dtype=torch.bool, device=env.device)':
        '        from scripts.native_recovery_query_bank_v1 import NativeRecoveryQueryBankV1\n'
        '        query_bank = NativeRecoveryQueryBankV1(env, native_replay.model, args.output)\n'
        '        finished = torch.zeros(count, dtype=torch.bool, device=env.device)',
        '                replay_state=None':
        '                query_bank.capture(tick, obs, action, finished)\n'
        '                replay_state=None',
        "    finally:\n        if 'isolation_observer' in locals():":
        "    finally:\n        if 'query_bank' in locals():\n            query_bank.save()\n"
        "        if 'isolation_observer' in locals():",
    }
    for before, after in substitutions.items():
        if text.count(before) != 1:
            raise ValueError('Changed capture lifecycle: ' + before)
        text = text.replace(before, after)
    ast.parse(text)
    output.mkdir(parents=True)
    entry = output / 'evaluate_query_bank.py'
    entry.write_text(text)
    files = [source_plan, source, Path(__file__).resolve(),
        root / 'scripts/native_recovery_query_bank_v1.py',
        root / 'scripts/audit_native_recovery_query_bank_v1.py']
    result = dict(entry=str(entry), entry_sha256=sha(entry),
        source_sha256={str(f): sha(f) for f in files},
        paired_run=plan['output'], controls_changed=False, scoring_changed=False,
        native_execution=False, observer_passivity_pending=True,
        control_ticks=[200, 799], lanes=128, maximum_buffer_bytes=64 * 1024**2)
    (output / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.root, args.output)))
