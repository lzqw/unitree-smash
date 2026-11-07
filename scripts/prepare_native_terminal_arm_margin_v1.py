"""Generate separately pinned native entry and strict audit for the corrected candidate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ENTRY_SHA='7331853475fad92d7cfce0d657b003b4ac8f998316279102aa9e6d9128b325ce'
AUDIT_SHA='9c089c1215e1c61f29ce42fec7e6f73e12b8b36d84ec0f5f36695d7b2a685d1e'


def prepare(entry_source,audit_source,output):
    entry_source=Path(entry_source);audit_source=Path(audit_source);output=Path(output)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(entry_source)!=ENTRY_SHA or sha(audit_source)!=AUDIT_SHA or output.exists():
        raise ValueError('Pinned source and new destination required')
    entry=entry_source.read_text();audit=audit_source.read_text()
    replacements=[
        ('audit','ROOT=Path(__file__).resolve().parents[1]',
         'ROOT=Path('+repr(str(Path(__file__).resolve().parents[1]))+')',1),
        ('entry','from scripts.native_terminal_support_v1 import NativeTerminalSupportV1 as NativeRecoverySequenceReplay',
         'from scripts.native_terminal_arm_margin_v1 import NativeTerminalArmMarginV1 as NativeRecoverySequenceReplay',1),
        ('audit','from scripts.native_terminal_support_v1 import load_verified_policy',
         'from scripts.native_terminal_arm_margin_v1 import load_verified_policy',1),
        ('entry','planned_prefix_terminal_feedback_v1','planned_prefix_terminal_arm_margin_v1',1),
        ('audit','planned_prefix_terminal_feedback_v1','planned_prefix_terminal_arm_margin_v1',2),
        ('audit','command,spheres,geometry=c.sequence_geometry)',
         'command,spheres,geometry=c.sequence_geometry,substeps=160)',1),
    ]
    values=dict(entry=entry,audit=audit)
    for kind,old,new,count in replacements:
        if values[kind].count(old)!=count:raise ValueError('Changed integration contract: '+old)
        values[kind]=values[kind].replace(old,new)
    for text in values.values():ast.parse(text)
    output.mkdir(parents=True)
    entry_path=output/'evaluate_terminal_arm_margin.py';audit_path=output/'audit_terminal_arm_margin.py'
    entry_path.write_text(values['entry']);audit_path.write_text(values['audit'])
    result=dict(entry=str(entry_path.resolve()),entry_sha256=sha(entry_path),
        audit=str(audit_path.resolve()),audit_sha256=sha(audit_path),
        entry_source=str(entry_source.resolve()),entry_source_sha256=ENTRY_SHA,
        audit_source=str(audit_source.resolve()),audit_source_sha256=AUDIT_SHA,
        preparer_sha256=sha(__file__),native_execution=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['entry-source','audit-source','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.entry_source,args.audit_source,args.output)))
