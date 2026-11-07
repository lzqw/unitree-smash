"""Prepare a separate storage-protected native entry for the restored planner."""
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
        ('from scripts.native_frictional_recovery_v1 import NativeFrictionalRecoveryV1 as NativeRecoverySequenceReplay',
         'from scripts.native_frictional_recovery_v2 import NativeFrictionalRecoveryV2 as NativeRecoverySequenceReplay'),
        ('recovery_sequence_case=args.recovery_sequence_case,',
         "recovery_sequence_case=args.recovery_sequence_case,\n        recovery_planner_version='local_restoration_v2',")]
    for before,after in replacements:
        if text.count(before)!=1:raise ValueError('Changed native integration contract')
        text=text.replace(before,after)
    ast.parse(text);output.mkdir(parents=True)
    entry=output/'evaluate_frictional_v2.py';entry.write_text(text)
    result=dict(source=str(source.resolve()),source_sha256=sha(source),entry=str(entry.resolve()),
        entry_sha256=sha(entry),preparer_sha256=sha(__file__),native_execution=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
