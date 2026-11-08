"""Measure airborne reference motion; do not infer native causal improvement."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh audit output required')
    paths=[a.run/'pure_control_commands.npz',a.run/'pure_control_diagnostics.jsonl']
    with np.load(paths[0]) as z:
        lane=z['lane'].copy();tick=z['tick'].copy();time=z['time_s'].copy()
    rows=[json.loads(x) for x in paths[1].read_text().splitlines()]
    if len(rows)!=len(lane):raise ValueError('Diagnostics not aligned with commands')
    previous={};speeds=[];case_rows={}
    for i,row in enumerate(rows):
        case=int(lane[i]);d=row['diagnostic'];ref=d['reference']
        feet=np.asarray(ref['swing_feet'],float)
        if feet.shape!=(4,3):raise ValueError('Unexpected foot reference shape')
        old=previous.get(case)
        if old is not None:
            j,last,last_phase=old;dt=float(time[i]-time[j])
            if tick[i]==tick[j]+1 and .009<dt<.011 and d['phase']==last_phase=='AIR_PREPARE':
                vz=(feet[:,2]-last[:,2])/dt
                speeds.extend(vz.tolist());case_rows.setdefault(case,[]).extend(vz.tolist())
        previous[case]=(i,feet,d['phase'])
    def stats(x):
        return dict(count=len(x),minimum=float(np.min(x)),median=float(np.median(x)),
            maximum=float(np.max(x)),fraction_below_minus_05=float(np.mean(np.asarray(x)<-.5))) if len(x) else None
    result=dict(reference_vertical_velocity_m_s=stats(speeds),
        cases={str(k):stats(v) for k,v in case_rows.items()},
        interpretation='A moving reference requires matching velocity/acceleration terms. These finite differences describe the goal, not actual foot motion or causal recovery gains.',
        no_new_native_execution=True,
        sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),*paths]})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['reference_vertical_velocity_m_s']))


if __name__=='__main__':main()
