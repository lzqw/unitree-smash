"""Prepare variable-length recovery diagnostics in an isolated native entry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_recovery_sequence_v1 import transform as original_transform


def transform(source,root):
    source=original_transform(source,root)
    old='from scripts.native_recovery_sequence_replay_v1 import NativeRecoverySequenceReplay'
    if source.count(old)!=1:raise ValueError('Unexpected native adapter import')
    source=source.replace(old,'from scripts.native_recovery_sequence_replay_v2 import NativeRecoverySequenceReplayV2 as NativeRecoverySequenceReplay')
    compile(source,'<variable-length-native-sequence>','exec')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous preparation')
    old=a.source.read_text();new=transform(old,ROOT);a.output.mkdir(parents=True)
    entry=a.output/'evaluate_sequence.py';entry.write_text(new)
    record=dict(live_entry_modified=False,source=str(a.source.resolve()),
        source_sha256=hashlib.sha256(old.encode()).hexdigest(),entry=str(entry.resolve()),
        entry_sha256=hashlib.sha256(new.encode()).hexdigest(),
        scope='Variable-length open-loop native response diagnostic, not measured-feedback MPC.')
    (a.output/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
