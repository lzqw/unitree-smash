"""Generate a separate pinned native entry without changing live experiments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256='f6df4a7e8e34b1e3161fde8e2e65e96f689281c79df53493ddb97390825bf25c'


def prepare(source,output):
    source=Path(source);output=Path(output)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(source)!=SOURCE_SHA256 or output.exists():raise ValueError('Pinned source and fresh output required')
    text=source.read_text()
    replacements=[
        ("parser.add_argument('--frictional-sphere-audit', type=Path, required=True)",
         "parser.add_argument('--frictional-sphere-audit', type=Path, required=True)\nparser.add_argument('--feedback-validation', type=Path, required=True)"),
        ('from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1 as NativeRecoverySequenceReplay',
         'from scripts.native_feedback_portfolio_v4 import NativeFeedbackPortfolioV4 as NativeRecoverySequenceReplay'),
        ('args.frictional_warm_plan,args.frictional_sphere_audit)',
         'args.frictional_warm_plan,args.frictional_sphere_audit,args.feedback_validation)'),
        ('recovery_sequence_case=args.recovery_sequence_case,',
         "recovery_sequence_case=args.recovery_sequence_case,\n        feedback_validation=str(args.feedback_validation.resolve()),\n        feedback_validation_sha256=digest(args.feedback_validation),\n        recovery_planner_version='verified_feedback_portfolio_v4',")]
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('Changed integration contract')
        text=text.replace(old,new)
    ast.parse(text);output.mkdir(parents=True)
    entry=output/'evaluate_feedback_portfolio.py';entry.write_text(text)
    record=dict(source=str(source.resolve()),source_sha256=sha(source),entry=str(entry.resolve()),
                entry_sha256=sha(entry),preparer_sha256=sha(__file__),native_execution=False)
    (output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
