"""Summarize accepted commands and moving foot goals before native failure.

Uses actual endpoint states and paired ground-force intervals. This audit
describes the old controller; it does not prove a counterfactual would recover.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','configuration','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cases',nargs='+',type=int,required=True)
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    if a.output.exists() or read(a.run/'job.json')['status']!='SUCCEEDED':raise ValueError('Fresh completed-run audit required')
    cfg=read(a.configuration);meta=read(a.run/'pure_control_ground_forces.json')
    c=PureControlReferenceClockV12(PositionPDInterface(**cfg['interface']),
        read(a.run/'native_command_inputs.json')['default_joint_position_rad'],cfg['model_reference_library'])
    with np.load(a.run/'pure_control_commands.npz') as z:rows={k:z[k].copy() for k in z.files}
    with np.load(a.run/'pure_control_ground_forces.npz') as z:peaks=z['interval_peak_normal_n'].copy()
    diagnostics=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics)!=len(rows['lane']):raise ValueError('Unaligned records')
    foot_ids=[c.model.names.index(n) for n in c.model.feet]
    nonfoot=[i for i,n in enumerate(meta['body_names']) if not n.endswith('_foot')]
    results=[]
    for lane in a.cases:
        ids=np.flatnonzero(rows['lane']==lane)
        active=[i for i in ids if diagnostics[int(i)]['diagnostic'].get('reference_active')]
        if not active:raise ValueError('No reference for requested case')
        begin=int(rows['tick'][active[-1]])+1
        owner_start=int(rows['tick'][ids[0]])
        touched=np.any(peaks[owner_start:,lane][:,nonfoot]>0.,axis=1)
        end=owner_start+int(np.flatnonzero(touched)[0]) if touched.any() else 800
        counts=Counter();events=[]
        for i in ids:
            i=int(i);t=int(rows['tick'][i]);r=diagnostics[i]
            if (r['tick'],r['lane'])!=(t,lane):raise ValueError('Unaligned row')
            if not begin<=t<end:continue
            d=r['diagnostic'];contacts=np.asarray(r['phase']['ground_contacts'],bool)
            counts['commands']+=1;counts['qp_accepted']+=int(d['solver_accepted'])
            counts[d['source']]+=1
            feet=np.asarray(c.model.poses(rows['q'][i])[0])[foot_ids]
            targets=np.asarray(d['reference']['swing_feet'])
            error=np.linalg.norm(targets-feet,axis=1)
            if (~contacts).any():
                k=int(np.argmax(np.where(~contacts,error,-1.)))
                events.append(dict(tick=t,foot=c.model.feet[k],soft_target_distance_m=float(error[k]),
                    contacts=contacts.tolist(),solver_accepted=bool(d['solver_accepted']),
                    actual_foot_world=feet[k].tolist(),target_world=targets[k].tolist()))
        results.append(dict(case=lane,reference_end_tick=begin,first_nonfoot_positive_force_tick=end if end<800 else None,
            counts=dict(counts),largest_noncontact_foot_targets=sorted(events,key=lambda r:r['soft_target_distance_m'],reverse=True)[:3]))
    files=[Path(__file__),a.configuration]+[a.run/n for n in ['job.json','native_command_inputs.json',
        'pure_control_commands.npz','pure_control_diagnostics.jsonl','pure_control_ground_forces.json','pure_control_ground_forces.npz']]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(results=results,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        actual_native_rerun=False,soft_target_distance_is_not_proof_of_error=True,valid_expert_labels=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(results))


if __name__=='__main__':main()
