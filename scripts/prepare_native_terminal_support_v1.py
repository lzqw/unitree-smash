"""Prepare an isolated entry for a whole-episode terminal-feedback diagnostic."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256='f6df4a7e8e34b1e3161fde8e2e65e96f689281c79df53493ddb97390825bf25c'


def prepare(source,output):
    source=Path(source);output=Path(output);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(source)!=SOURCE_SHA256 or output.exists():raise ValueError('Pinned source and new output required')
    text=source.read_text()
    replacements=[
        ("parser.add_argument('--frictional-sphere-audit', type=Path, required=True)",
         "parser.add_argument('--frictional-sphere-audit', type=Path, required=True)\nparser.add_argument('--terminal-support-validation', type=Path, required=True)"),
        ('from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1 as NativeRecoverySequenceReplay',
         'from scripts.native_terminal_support_v1 import NativeTerminalSupportV1 as NativeRecoverySequenceReplay'),
        ('args.frictional_warm_plan,args.frictional_sphere_audit)',
         'args.frictional_warm_plan,args.frictional_sphere_audit,args.terminal_support_validation)'),
        ('recovery_sequence_case=args.recovery_sequence_case,',
         "recovery_sequence_case=args.recovery_sequence_case,\n        terminal_support_validation=str(args.terminal_support_validation.resolve()),\n        terminal_support_validation_sha256=digest(args.terminal_support_validation),\n        recovery_controller_mode='planned_prefix_terminal_feedback_v1',")]
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('Changed native integration contract')
        text=text.replace(old,new)
    ast.parse(text);output.mkdir(parents=True);entry=output/'evaluate_terminal_support.py';entry.write_text(text)
    result=dict(source=str(source.resolve()),source_sha256=sha(source),entry=str(entry.resolve()),
        entry_sha256=sha(entry),preparer_sha256=sha(__file__),native_execution=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
