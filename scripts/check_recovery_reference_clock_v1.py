"""Verify reference indices against recorded native control ticks and float32 clocks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.pure_control_long_trajectory_recovery_v11 import PureControlLongTrajectoryRecoveryV11


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh check output required')
    with np.load(args.native_run/'pure_control_commands.npz') as z:
        ticks=z['tick'].copy();lanes=z['lane'].copy();times=z['time_s'].copy()
    old=PureControlLongTrajectoryRecoveryV11.__new__(PureControlLongTrajectoryRecoveryV11)
    new=PureControlReferenceClockV12.__new__(PureControlReferenceClockV12)
    total=0;old_errors=0;examples=[]
    for length in [60,160,320]:
        old.chosen=new.chosen={'trajectory':[None]*length}
        for lane in sorted(set(lanes.tolist())):
            ids=np.flatnonzero(lanes==lane)
            t0=float(times[ids[0]]);k0=int(ticks[ids[0]])
            for i in ids:
                k=int(ticks[i])-k0
                expected=k if k<length else None
                old.elapsed=new.elapsed=float(times[i])-t0
                oi,_=old._reference_time();ni,_=new._reference_time()
                if ni!=expected:raise ValueError('Corrected index disagrees with native tick')
                total+=1
                if oi!=expected:
                    old_errors+=1
                    if len(examples)<8:examples.append(dict(case=int(lane),length=length,tick=k,elapsed=new.elapsed,old=oi,new=ni))
    synthetic=0
    for start in [0.,2.70,2.75,2.82,100.]:
        for cast in [float,np.float32]:
            for length in [60,160,320]:
                new.chosen={'trajectory':[None]*length}
                for k in range(801):
                    new.elapsed=float(cast(start+.01*k))-float(cast(start))
                    expected=k if k<length else None
                    if new._reference_time()[0]!=expected:raise ValueError('Synthetic clock mismatch')
                    synthetic+=1
    new.chosen={'trajectory':[None]*60};new.elapsed=.004
    try:new._reference_time()
    except ValueError:pass
    else:raise ValueError('Non-grid caller should be rejected')
    result=dict(passed=True,native_clock_checks=total,old_index_mismatches=old_errors,
        corrected_index_mismatches=0,synthetic_checks=synthetic,examples=examples,
        native_success=False,physical_limits_unchanged=True,
        source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'scripts/pure_control_reference_clock_v12.py',args.native_run/'pure_control_commands.npz']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
