"""Add bounded passive telemetry to a separate pinned feedback-MPC entry."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = '78734df8dc2b16132418faa93a9a6f2ea60c0ee39d205bfd050d56f87cd9e9d4'


def prepare(source, output):
    source=Path(source);output=Path(output)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(source)!=SOURCE_SHA256 or output.exists():
        raise ValueError('Pinned source and fresh directory required')
    text=source.read_text()
    old='        sequence_student_commands=[];sequence_final_commands=[];sequence_first_episode=[]'
    new='''        from scripts.native_contact_microtrace_v1 import NativeContactMicrotraceV1
        micro_steps = int(round(protocol['policy_dt_s'] / env.physics_dt))
        if abs(micro_steps * env.physics_dt - protocol['policy_dt_s']) > 1e-12:
            raise ValueError('Noninteger control/physics ratio')
        contact_microtrace = NativeContactMicrotraceV1(env, args.output,
            lane=args.recovery_sequence_case,
            first_frame=native_replay.start * micro_steps,
            last_frame=(native_replay.start + 20) * micro_steps)
''' + old
    end='''    finally:
        if 'native_replay' in locals():
            native_replay.planner.close()
        env._diagnostic_observer=None
        env.close()'''
    ending='''    finally:
        try:
            if 'contact_microtrace' in locals():
                try:
                    contact_microtrace.save()
                finally:
                    contact_microtrace.detach()
        finally:
            if 'native_replay' in locals():
                native_replay.planner.close()
            env._diagnostic_observer=None
            env.close()'''
    for before,after in [(old,new),(end,ending)]:
        if text.count(before)!=1:
            raise ValueError('Changed entry integration contract')
        text=text.replace(before,after)
    ast.parse(text)
    output.mkdir(parents=True)
    entry=output/'evaluate_feedback_contact_microtrace.py'
    entry.write_text(text)
    record=dict(source=str(source.resolve()),source_sha256=sha(source),
                entry=str(entry.resolve()),entry_sha256=sha(entry),
                preparer_sha256=sha(__file__),native_execution=False,
                queued=False,physics_or_controller_changed=False,
                observation_window_control_steps=20)
    (output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output)))
