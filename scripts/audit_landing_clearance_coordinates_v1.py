#!/usr/bin/env python3
"""Compare legacy and corrected clearance QPs on recorded first-episode states.

CPU counterfactual only: neither predicted feasibility nor target changes
constitute native recovery. All sampled post-touchdown states are retained.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np

from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.reduced_whole_body_support_qp_v1 import ReducedWholeBodySupportQP
from scripts.whole_body_support_qp_v1 import PositionPDInterface, SupportSettings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--legacy-ref',default='55534de')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing audit')
    root=Path(__file__).resolve().parents[1]
    old_source=subprocess.check_output(['git','show',args.legacy_ref+':scripts/reduced_whole_body_support_qp_v1.py'],cwd=root,text=True)
    legacy=types.ModuleType('legacy_clearance_qp')
    exec(compile(old_source,'legacy_clearance_qp','exec'),legacy.__dict__)
    native=json.loads((args.source/'native_command_inputs.json').read_text())
    default=np.array(native['default_joint_position_rad']);limits=np.array(native['joint_limits_rad'])
    interface=PositionPDInterface(limits,np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    settings=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    old=legacy.ReducedWholeBodySupportQP(interface,settings)
    new=ReducedWholeBodySupportQP(interface,settings)
    records=[];counts=Counter();sources={}
    for case in (1,67,95):
        path=args.source/'fixed_policy_traces'/f'case_{case:03d}.npz'
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as trace:
            public=unpack_public(trace['packed_public']);actions=trace['dispatched_actions']
        for tick in np.flatnonzero(public[:,774]>.5)[::10]:
            state=new.model.state_from_public(public[tick],default)
            if not state.contacts.any():continue
            target=limits[:,0]+.5*(actions[tick]+1.)*(limits[:,1]-limits[:,0])
            target[:12]=.85*target[:12]+.15*default[:12]
            measured=old.model.snapshot(state.q,state.v)
            old.com_kinematics(state);old.reference(state,measured,target)
            z_errors={name:float(old.model.data.oMf[old.model.body_frames[name]].translation[2]-
                                measured['body_positions_world'][old.model.names.index(name),2])
                      for name in ('FL_thigh','FR_thigh','RL_thigh','RR_thigh')}
            before=old.solve(state,target);after=new.solve(state,target)
            counts['samples']+=1;counts['legacy_accepted']+=bool(before['accepted']);counts['corrected_accepted']+=bool(after['accepted'])
            both=before['accepted'] and after['accepted']
            delta=float(np.max(np.abs(after['joint_position_target']-before['joint_position_target']))) if both else None
            records.append(dict(case=case,tick=int(tick),clock_s=float(public[tick,775]),
                contacts=state.contacts.tolist(),legacy_height_error_m=z_errors,
                legacy_accepted=bool(before['accepted']),corrected_accepted=bool(after['accepted']),
                legacy_status=before.get('solver_status'),corrected_status=after.get('solver_status'),
                common_accepted_target_difference_max_rad=delta))
    errors=[abs(x) for r in records for x in r['legacy_height_error_m'].values()]
    out=dict(scope='CPU same-state QP counterfactual on recorded public observations; not a native performance result',
        source=str(args.source),legacy_ref=args.legacy_ref,legacy_qp_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
        corrected_qp_sha256=hashlib.sha256((root/'scripts/reduced_whole_body_support_qp_v1.py').read_bytes()).hexdigest(),
        recorded_input_sha256=sources,sampling='every tenth phase774 first-episode row for TRAIN1/67/95',
        counts=dict(counts),legacy_absolute_height_error_m=dict(median=float(np.median(errors)),maximum=max(errors)),records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='records'}))


if __name__=='__main__':main()
