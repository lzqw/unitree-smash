"""Add prefix evidence checkpoints to an unstarted microtrace evaluator."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256='99f239d14bf9e3d7b98765606d4b5c183d9884b610df6c87068b613b8a24d88f'


def prepare(source,output):
    source=Path(source);output=Path(output)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(source)!=SOURCE_SHA256 or output.exists():raise ValueError('Pinned input and fresh output required')
    text=source.read_text()
    changes=[
        ('        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]',
         '        from scripts.recovery_partial_evidence_v1 import save_partial_recovery\n'
         '        recovery_completed_control_ticks=0\n'
         '        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]'),
        ('                observation, _, done, _ = wrapped.step(action)',
         '                observation, _, done, _ = wrapped.step(action)\n'
         '                recovery_completed_control_ticks=tick+1\n'
         '                if tick>=native_replay.start and (tick+1)%10==0:\n'
         '                    save_partial_recovery(native_replay,recovery_completed_control_ticks)'),
        ('''        try:
            if 'contact_microtrace' in locals():
                try:
                    contact_microtrace.save()
                finally:
                    contact_microtrace.detach()''',
         '''        try:
            try:
                if 'native_replay' in locals() and 'recovery_completed_control_ticks' in locals():
                    save_partial_recovery(native_replay,recovery_completed_control_ticks)
            finally:
                if 'contact_microtrace' in locals():
                    try:
                        contact_microtrace.save()
                    finally:
                        contact_microtrace.detach()''')]
    for old,new in changes:
        if text.count(old)!=1:raise ValueError('Changed integration point')
        text=text.replace(old,new)
    ast.parse(text);output.mkdir(parents=True)
    entry=output/'evaluate_feedback_recorded.py';entry.write_text(text)
    result=dict(source=str(source.resolve()),source_sha256=sha(source),entry=str(entry.resolve()),
                entry_sha256=sha(entry),preparer_sha256=sha(__file__),native_execution=False,
                full_run_admission_changed=False,control_or_physics_changed=False)
    (output/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
