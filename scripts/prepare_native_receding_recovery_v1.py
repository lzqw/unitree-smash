"""Prepare, but do not launch, an isolated measured-feedback native entry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_recovery_sequence_v2 import transform as replay_transform


def transform(source):
    result=replay_transform(source,ROOT)
    old='from scripts.native_recovery_sequence_replay_v2 import NativeRecoverySequenceReplayV2 as NativeRecoverySequenceReplay'
    if result.count(old)!=1:raise ValueError('Changed replay adapter import')
    result=result.replace(old,'from scripts.native_receding_recovery_v1 import NativeRecedingRecoveryV1 as NativeRecoverySequenceReplay')
    old='deployable_risk_trigger=False,sustained_mpc_feedback_implemented=False,'
    if result.count(old)!=1:raise ValueError('Changed diagnostic summary')
    result=result.replace(old,'deployable_risk_trigger=False,sustained_mpc_feedback_implemented=True,\n'
        '            teacher_steps=sequence_result["teacher_steps"],\n'
        '            unverified_student_fallback_steps=sequence_result["unverified_student_fallback_steps"],')
    compile(result,'<native-feedback-entry>','exec')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous prepared entry')
    old=a.source.read_text();new=transform(old);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_feedback.py';entry.write_text(new)
    record=dict(source=str(a.source.resolve()),source_sha256=hashlib.sha256(old.encode()).hexdigest(),
        entry=str(entry.resolve()),entry_sha256=hashlib.sha256(new.encode()).hexdigest(),
        live_entry_modified=False,native_execution=False,gpu_job_started=False,
        scope='Prepared case/time diagnostic of measured-state replanning, not yet native validated.')
    (a.output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
