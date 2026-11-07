"""Prepare a separate native entry and full audit for explicit feedback tails."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ENTRY_SHA = '988ffbd818cae464c1b72bca22be23966c89586b736ad7ca3f4923269f63a3c4'
AUDIT_SHA = 'e7f3dca3b74b853c52b8a56d64bfd3f48e5186dd39f4bec4b515625a7db69fa1'


def prepare(entry_source, audit_source, output):
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    entry_source, audit_source, output = map(Path, [entry_source, audit_source, output])
    if output.exists() or sha(entry_source) != ENTRY_SHA or sha(audit_source) != AUDIT_SHA:
        raise ValueError('Pinned inputs and fresh destination required')
    entry, audit = entry_source.read_text(), audit_source.read_text()
    replacements = [
        ('from scripts.native_feedback_portfolio_v4 import NativeFeedbackPortfolioV4 as NativeRecoverySequenceReplay',
         'from scripts.native_feedback_tail_v1 import NativeFeedbackTailV1 as NativeRecoverySequenceReplay'),
        ("recovery_planner_version='verified_feedback_portfolio_v4'", "recovery_planner_version='feedback_tail_candidate_selection_v1'")]
    for old, new in replacements:
        if entry.count(old) != 1:
            raise ValueError('Changed native integration point')
        entry = entry.replace(old, new)
    root = str(Path(__file__).resolve().parents[1])
    old = 'ROOT=Path(__file__).resolve().parents[1]'
    if audit.count(old) != 1:
        raise ValueError('Changed audit root')
    audit = audit.replace(old, 'ROOT=Path('+repr(root)+')')
    marker = "    for j,plan in enumerate(result['planning_records']):"
    if audit.count(marker) != 1:
        raise ValueError('Changed audit loop')
    audit = audit.replace(marker,
        '    from scripts.audit_feedback_tail_plan_v1 import load_audit_policy, check_plan\n'
        '    policy, protocol = load_audit_policy(c, result, inputs, a.bundle, manifest)\n\n'+marker)
    start = audit.index('        friction_core.CONTEXT=(c,state,state.previous_command_target,commands,spheres,c.sequence_geometry)')
    end = audit.index("        end=result['planning_records'][j+1]", start)
    audit = audit[:start] + (
        '        checked, prediction = check_plan(c, state, commands, spheres, policy, protocol, plan)\n\n'
    ) + audit[end:]
    marker = "    report['source_sha256']={str(f.resolve()):sha(f) for f in files}"
    if audit.count(marker) != 1:
        raise ValueError('Changed audit provenance')
    audit = audit.replace(marker,
        "    files += [Path(result['feedback_validation']), ROOT/'configs/evaluation_protocol_v2.json']\n"
        "    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]\n"
        "    report['planner_version'] = result['planner_version']\n"
        "    report['acceptance_rule'] = result['acceptance_rule']\n"+marker)
    for text in [entry, audit]:
        ast.parse(text)
    output.mkdir(parents=True)
    entry_path = output/'evaluate_feedback_tail.py'; audit_path = output/'audit_feedback_tail.py'
    entry_path.write_text(entry); audit_path.write_text(audit)
    record = dict(entry=str(entry_path.resolve()), entry_sha256=sha(entry_path),
        audit=str(audit_path.resolve()), audit_sha256=sha(audit_path),
        entry_source=str(entry_source.resolve()), entry_source_sha256=ENTRY_SHA,
        audit_source=str(audit_source.resolve()), audit_source_sha256=AUDIT_SHA,
        preparer_sha256=sha(__file__), native_execution=False, queued=False,
        strict_panel_tolerances_changed=False, final_command_mapping_changed=False,
        executed_prefix_checkpointing_retained=True, microtrace_retained=True)
    (output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['entry-source', 'audit-source', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.entry_source, a.audit_source, a.output)))
