"""Clone the pinned terminal-feedback entry into an explicit nonexecuting control."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = '7331853475fad92d7cfce0d657b003b4ac8f998316279102aa9e6d9128b325ce'


def prepare(source, output):
    source, output = Path(source), Path(output)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(source) != SOURCE_SHA256 or output.exists():
        raise ValueError('Require unchanged terminal entry and a new preparation directory')
    text = source.read_text()
    replacements = [
        ('from scripts.native_terminal_support_v1 import NativeTerminalSupportV1 as NativeRecoverySequenceReplay',
         'from scripts.native_terminal_shadow_v1 import NativeTerminalShadowV1 as NativeRecoverySequenceReplay'),
        ("recovery_controller_mode='planned_prefix_terminal_feedback_v1',",
         "recovery_controller_mode='terminal_feedback_shadow_not_executed_v1',\n        native_shadow_only=True,"),
        ('case=native_replay.case,applied_steps=native_replay.applied,',
         'case=native_replay.case,applied_steps=0,\n            shadow_proposal_steps=native_replay.shadow_steps,native_shadow_only=True,'),
        ('deployable_risk_trigger=False,sustained_mpc_feedback_implemented=True,',
         'deployable_risk_trigger=False,sustained_mpc_feedback_implemented=False,')]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError('Changed pinned integration contract: ' + before)
        text = text.replace(before, after)
    ast.parse(text)
    output.mkdir(parents=True)
    entry = output / 'evaluate_terminal_shadow.py'
    entry.write_text(text)
    record = dict(source=str(source.resolve()), source_sha256=sha(source),
        entry=str(entry.resolve()), entry_sha256=sha(entry), preparer_sha256=sha(__file__),
        native_execution=False, shadow_proposals_are_not_execution=True,
        valid_expert_labels=False, actual_native_equivalence_unverified=True)
    (output / 'preparation.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output)))
