#!/usr/bin/env python3
"""Compare held-command predictions with the next recorded native state.

Uses actual executed targets, not hindsight control selection. Samples every
20th recorded first-episode tick. Contact changes and bound violations remain
visible; better joint prediction alone is not a valid support certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),maximum=float(a.max())) if len(a) else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier audit')
    read=lambda f:json.loads((args.run/f).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Require completed native run')
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    limits=np.asarray(native['joint_limits_rad'])
    interface=PositionPDInterface(limits,np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    mpc=LandingCommandHistoryMPCV1(interface,qp_settings=SupportSettings(),
        settings=LandingMPCSettingsV2(executor_replay=True),arm_target_rate_rad_s=10.)
    logs=read('landing_mpc_records.json');cache={};hashes={};rows=[]
    for record in logs:
        if record.get('episode_index')!=0 or record.get('control_tick',-1)%20:continue
        case=record['case'];tick=record['control_tick']
        if case not in cache:
            path=args.run/'fixed_policy_traces'/f'case_{case:03d}.npz'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as z:cache[case]=unpack_public(z['packed_public'])
        public=cache[case]
        if tick+1>=len(public):continue
        if abs(float(public[tick+1,775]-public[tick,775])-.01)>2e-6:raise ValueError('Not an adjacent native step')
        state=mpc.qp.model.state_from_public(public[tick],default)
        actual=mpc.qp.model.state_from_public(public[tick+1],default)
        state.contacts=np.asarray(record['support_contacts'],dtype=bool)
        state.previous_command_target=np.asarray(record['previous_dispatched_target'])
        target=np.asarray(record['final_dispatched_joint_target_rad'])
        torque=np.clip(interface.kp*(target-state.joint_position)-interface.kd*state.joint_velocity,
                       -interface.effort,interface.effort)
        row=dict(case=case,tick=tick,applied=record['applied_first_step'],
            observed_contact_change=not np.array_equal(public[tick,57:61]>.5,public[tick+1,57:61]>.5),models={})
        for name,steps in [('held_initial_torque',0),('implicit_1',1),('implicit_16',16),('implicit_160',160)]:
            started=time.perf_counter()
            try:
                if not steps:
                    predicted=mpc._integrate(state,np.zeros(24),dict(torque=torque,joint_position_target=target))
                    diagnostics=None
                else:
                    result=predict_implicit_support(mpc.qp.model,interface,state,target,substeps=steps)
                    predicted=result['state'];diagnostics=result['diagnostics']
                qerr=predicted.joint_position-actual.joint_position
                verr=predicted.joint_velocity-actual.joint_velocity
                row['models'][name]=dict(ok=True,runtime_ms=1000*(time.perf_counter()-started),
                    leg_position_error_max_rad=float(np.max(abs(qerr[:12]))),
                    arm_position_error_max_rad=float(np.max(abs(qerr[12:]))),
                    leg_velocity_error_max_rad_s=float(np.max(abs(verr[:12]))),
                    arm_velocity_error_max_rad_s=float(np.max(abs(verr[12:]))),
                    base_position_error_m=float(np.linalg.norm(predicted.q[:3]-actual.q[:3])),
                    base_linear_velocity_error_mps=float(np.linalg.norm(predicted.v[:3]-actual.v[:3])),
                    base_angular_velocity_error_rad_s=float(np.linalg.norm(predicted.v[3:6]-actual.v[3:6])),
                    physical_model_diagnostics=diagnostics)
            except (ValueError,np.linalg.LinAlgError) as error:
                row['models'][name]=dict(ok=False,error=repr(error))
        rows.append(row)
    keys=['leg_position_error_max_rad','arm_position_error_max_rad','leg_velocity_error_max_rad_s',
          'arm_velocity_error_max_rad_s','base_position_error_m','base_linear_velocity_error_mps',
          'base_angular_velocity_error_rad_s','runtime_ms']
    summaries={}
    for name in ['held_initial_torque','implicit_1','implicit_16','implicit_160']:
        subset=[r['models'][name] for r in rows if r['models'][name]['ok']]
        summaries[name]=dict(successful_predictions=len(subset),failed_predictions=len(rows)-len(subset),
            errors={k:stats([x[k] for x in subset]) for k in keys})
    for name in ['landing_mpc_records.json','native_command_inputs.json','job.json']:
        hashes[str(args.run/name)]=hashlib.sha256((args.run/name).read_bytes()).hexdigest()
    for path in [Path(__file__),ROOT/'scripts/implicit_support_prediction_v1.py']:
        hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(scope=__doc__,native_new_execution=False,source=str(args.run),samples=len(rows),
        contact_changed_samples=sum(r['observed_contact_change'] for r in rows),summary=summaries,records=rows,source_sha256=hashes)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(samples=len(rows),summary=summaries)))


if __name__=='__main__':main()
