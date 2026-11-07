#!/usr/bin/env python3
"""Compare bilateral and unilateral implicit-PD predictions on actual commands.

10 ms predictions; no new native run, no controller score or safety claim.
Only the initial observed feet are contact candidates. Summaries compare the
same successful sample intersection and report all rejected predictions.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pinocchio as pin
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.audit_relinearized_support_prediction_v1 import refreshed
from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=100)
    a=p.parse_args()
    if a.output.exists() or a.stride<1:raise ValueError('Use a new output and positive stride')
    read=lambda n:json.loads((a.run/n).read_text())
    assert read('job.json')['status']=='SUCCEEDED'
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    model=LoadedPinocchioModelV1();cache={};rows=[];sources=[Path(__file__)]
    for r in read('landing_mpc_records.json'):
        if r.get('episode_index')!=0 or r['control_tick']%a.stride or not r['applied_first_step']:continue
        case,tick=r['case'],r['control_tick']
        if case not in cache:
            path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';sources.append(path)
            with np.load(path,allow_pickle=False) as z:cache[case]=unpack_public(z['packed_public'])
        public=cache[case]
        if tick+1>=len(public):continue
        assert abs(float(public[tick+1,775]-public[tick,775])-.01)<2e-6
        state=model.state_from_public(public[tick],default)
        actual=model.state_from_public(public[tick+1],default)
        state.contacts=np.asarray(r['support_contacts'],dtype=bool)
        target=np.asarray(r['final_dispatched_joint_target_rad'])
        row=dict(case=case,tick=tick,contact_observation_changed=not np.array_equal(
            public[tick,57:61]>.5,public[tick+1,57:61]>.5),models={})
        for name in ['bilateral_refreshed','unilateral']:
            start=time.perf_counter()
            try:
                if name=='bilateral_refreshed':
                    q,v,diagnostics=refreshed(model,interface,state,target)
                else:
                    out=predict_unilateral_support(model,interface,state,target)
                    q,v,diagnostics=out['state'].q,out['state'].v,out['diagnostics']
                R=pin.Quaternion(q[3:7]).matrix();Ra=actual.root_rotation_world
                row['models'][name]=dict(ok=True,
                    root_position_m=float(np.linalg.norm(q[:3]-actual.q[:3])),
                    root_attitude_rad=float(np.linalg.norm(pin.log3(Ra.T@R))),
                    world_linear_velocity_mps=float(np.linalg.norm(R@v[:3]-Ra@actual.v[:3])),
                    world_angular_velocity_rad_s=float(np.linalg.norm(R@v[3:6]-Ra@actual.v[3:6])),
                    leg_position_rad=float(np.max(abs(q[model.q_indices[:12]]-actual.joint_position[:12]))),
                    arm_position_rad=float(np.max(abs(q[model.q_indices[12:]]-actual.joint_position[12:]))),
                    diagnostics=diagnostics,runtime_ms=1000*(time.perf_counter()-start))
            except (ValueError,np.linalg.LinAlgError) as error:
                row['models'][name]=dict(ok=False,error=repr(error),runtime_ms=1000*(time.perf_counter()-start))
        rows.append(row)
        if len(rows)%25==0:print(json.dumps(dict(samples=len(rows))),flush=True)
    common=[r for r in rows if all(v['ok'] for v in r['models'].values())]
    keys=['root_position_m','root_attitude_rad','world_linear_velocity_mps',
          'world_angular_velocity_rad_s','leg_position_rad','arm_position_rad']
    def stats(values):
        return dict(mean=float(np.mean(values)),p95=float(np.quantile(values,.95)),
                    maximum=float(max(values))) if values else None
    summary={}
    for name in ['bilateral_refreshed','unilateral']:
        allvalid=[r['models'][name] for r in rows if r['models'][name]['ok']]
        summary[name]=dict(successful=len(allvalid),failed=len(rows)-len(allvalid),
            failure_reasons=dict(Counter(r['models'][name]['error'] for r in rows if not r['models'][name]['ok'])),
            common_sample_errors={k:stats([r['models'][name][k] for r in common]) for k in keys},
            runtime_ms=stats([r['models'][name]['runtime_ms'] for r in rows]))
    comparison={k:dict(unilateral_better=sum(r['models']['unilateral'][k]<r['models']['bilateral_refreshed'][k] for r in common),
        common_samples=len(common)) for k in keys}
    sources += [a.run/n for n in ['job.json','native_command_inputs.json','landing_mpc_records.json']]
    sources += [ROOT/'scripts'/n for n in ['unilateral_implicit_prediction_v1.py',
        'implicit_support_prediction_v1.py','audit_relinearized_support_prediction_v1.py',
        'loaded_pinocchio_model_v1.py','whole_body_support_qp_v1.py']]
    result=dict(scope=__doc__,source=str(a.run),native_new_execution=False,samples=len(rows),
        stride=a.stride,common_samples=len(common),summary=summary,paired_error_comparison=comparison,
        unilateral_released_samples=sum(r['models']['unilateral']['diagnostics']['released_foot_substeps']>0
            for r in common),
        unilateral_sliding_samples=sum(r['models']['unilateral']['diagnostics']['sliding_foot_substeps']>0
            for r in common),
        records=rows,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']}))


if __name__=='__main__':main()
