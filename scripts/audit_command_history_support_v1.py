#!/usr/bin/env python3
"""Compare support QPs on recorded states with actual previous final commands."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.admitted_arm_support_qp_v1 import AdmittedArmSupportQP
from scripts.command_history_support_qp_v1 import CommandHistorySupportQP
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm-target-rate',type=float,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve existing audit')
    n=json.loads((a.source/'native_command_inputs.json').read_text())
    default=np.array(n['default_joint_position_rad']);limits=np.array(n['joint_limits_rad'])
    interface=PositionPDInterface(limits,np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    cfg=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    old=AdmittedArmSupportQP(interface,cfg,preserve_arm_command=True)
    new=CommandHistorySupportQP(interface,cfg,preserve_arm_command=True,arm_target_rate_rad_s=a.arm_target_rate)
    records=[];hashes={}
    for case in (1,67,95):
        path=a.source/'fixed_policy_traces'/f'case_{case:03d}.npz'
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as trace:
            public=unpack_public(trace['packed_public']);actions=trace['dispatched_actions']
        for tick in np.flatnonzero(public[:,774]>.5)[::10]:
            if tick==0:continue
            state=new.model.state_from_public(public[tick],default)
            if not state.contacts.any():continue
            state.previous_command_target=limits[:,0]+.5*(actions[tick-1]+1.)*(limits[:,1]-limits[:,0])
            target=limits[:,0]+.5*(actions[tick]+1.)*(limits[:,1]-limits[:,0])
            target[:12]=.85*target[:12]+.15*default[:12]
            old.reset();new.reset();before=old.solve(state,target);after=new.solve(state,target)
            records.append(dict(case=case,tick=int(tick),clock_s=float(public[tick,775]),
                measured_center_accepted=before['accepted'],history_accepted=after['accepted'],
                history_status=after.get('solver_status'),history_rejection=after.get('rejection'),
                history_residuals=after.get('residuals'),
                previous_command_actual_tracking_error_max_rad=float(np.max(abs(state.previous_command_target-state.joint_position)))))
    counts=dict(samples=len(records),measured_center_accepted=sum(r['measured_center_accepted'] for r in records),
                command_history_accepted=sum(r['history_accepted'] for r in records),
                gained=sum(not r['measured_center_accepted'] and r['history_accepted'] for r in records),
                lost=sum(r['measured_center_accepted'] and not r['history_accepted'] for r in records))
    for name in ('whole_body_support_qp_v1.py','reduced_whole_body_support_qp_v1.py','command_history_support_qp_v1.py'):
        path=ROOT/'scripts'/name;hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    out=dict(scope='CPU same-state counterfactual, reference reset per sample, no native recovery claim',
        arm_target_rate_rad_s=a.arm_target_rate,physical_velocity_limits_rad_s=interface.velocity.tolist(),
        effort_limits_nm=interface.effort.tolist(),source_sha256=hashes,counts=counts,records=records)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(counts))

if __name__=='__main__':main()
