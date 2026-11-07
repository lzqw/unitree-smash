#!/usr/bin/env python3
"""CPU counterfactual for prescribed arm goals versus command admission.

Run from the repository root. This resets reference memory per sample and does
not claim native recovery or justify relaxing actuator/physics constraints.
"""
import argparse
import sys
ROOT=__import__('pathlib').Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json,hashlib
from pathlib import Path
import numpy as np
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.continuous_support_qp_v1 import ContinuousSupportQP
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=ROOT;source=args.source
if args.output.exists():raise RuntimeError('Preserve existing audit')
n=json.loads((source/'native_command_inputs.json').read_text());default=np.array(n['default_joint_position_rad']);limits=np.array(n['joint_limits_rad'])
i=PositionPDInterface(limits,np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
cfg=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
fixed=ContinuousSupportQP(i,cfg,preserve_arm_command=True);free=ContinuousSupportQP(i,cfg,preserve_arm_command=False)
admitted_fixed=ContinuousSupportQP(i,cfg,preserve_arm_command=True)
records=[]
for case in (1,67,95):
 with np.load(source/'fixed_policy_traces'/f'case_{case:03d}.npz') as tr:
  public=unpack_public(tr['packed_public']);actions=tr['dispatched_actions']
 for tick in np.flatnonzero(public[:,774]>.5)[::10]:
  state=fixed.model.state_from_public(public[tick],default)
  if not state.contacts.any():continue
  target=limits[:,0]+.5*(actions[tick]+1.)*(limits[:,1]-limits[:,0]);target[:12]=.85*target[:12]+.15*default[:12]
  fixed.reset();free.reset()
  a=fixed.solve(state,target);b=free.solve(state,target)
  admitted=state.joint_position+np.clip(target-state.joint_position,-i.velocity*.01,i.velocity*.01)
  admitted_target=target.copy();admitted_target[12:]=admitted[12:]
  admitted_fixed.reset();c=admitted_fixed.solve(state,admitted_target)
  torque_raw=np.clip(i.kp*(target-state.joint_position)-i.kd*state.joint_velocity,-i.effort,i.effort)
  torque_admit=np.clip(i.kp*(admitted-state.joint_position)-i.kd*state.joint_velocity,-i.effort,i.effort)
  records.append(dict(case=case,tick=int(tick),clock_s=float(public[tick,775]),fixed_accepted=a['accepted'],free_accepted=b['accepted'],admitted_fixed_accepted=c['accepted'],admitted_fixed_rejection=c.get('rejection'),admitted_fixed_status=c.get('solver_status'),fixed_status=a.get('solver_status'),free_status=b.get('solver_status'),fixed_rejection=a.get('rejection'),fixed_residuals=a.get('residuals'),free_rejection=b.get('rejection'),arm_target_admission_difference_rad=float(np.max(np.abs(target[12:]-admitted[12:]))),arm_torque_admission_difference_nm=float(np.max(np.abs(torque_raw[12:]-torque_admit[12:])))))
counts=dict(admitted_fixed_accepted=sum(r['admitted_fixed_accepted'] for r in records),samples=len(records),fixed_accepted=sum(r['fixed_accepted'] for r in records),free_accepted=sum(r['free_accepted'] for r in records),fixed_fail_free_pass=sum(not r['fixed_accepted'] and r['free_accepted'] for r in records),arm_admission_mismatch=sum(r['arm_target_admission_difference_rad']>1e-6 for r in records))
out=dict(scope='Same recorded public states, reset reference per sample; CPU feasibility diagnostic only, no native recovery or policy benefit.',source=str(source),counts=counts,records=records)
out['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'native_command_inputs.json',*[source/'fixed_policy_traces'/f'case_{c:03d}.npz' for c in (1,67,95)],root/'scripts/continuous_support_qp_v1.py',root/'scripts/reduced_whole_body_support_qp_v1.py',root/'scripts/whole_body_support_qp_v1.py']}
out['sampling']='Every tenth phase774 row in the first scored episode for TRAIN1/67/95, original limits, 0.01s admission.'
p=args.output
p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():raise RuntimeError('Preserve existing audit')
p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(counts))
