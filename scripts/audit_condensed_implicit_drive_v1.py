#!/usr/bin/env python3
"""Check substep constraint maps on actual commands from a completed trial."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.condensed_implicit_drive_v1 import condense_implicit_drive,interval_constraints
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audit')
    read=lambda f:json.loads((a.run/f).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Completed native source required')
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    model=LoadedPinocchioModelV1();S=np.zeros((24,18));S[model.v_indices,np.arange(18)]=1.
    cache={};rows=[];paths=[Path(__file__),ROOT/'scripts/condensed_implicit_drive_v1.py',
        ROOT/'scripts/implicit_support_prediction_v1.py']
    for record in read('landing_mpc_records.json'):
        tick=record['control_tick'];case=record['case']
        if tick%20 or not record.get('applied_first_step'):continue
        if case not in cache:
            path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';paths.append(path)
            with np.load(path,allow_pickle=False) as z:cache[case]=unpack_public(z['packed_public'])
        if tick+1>=len(cache[case]):continue
        state=model.state_from_public(cache[case][tick],default)
        state.contacts=np.asarray(record['support_contacts'],dtype=bool)
        target=np.asarray(record['final_dispatched_joint_target_rad'])
        snap=model.snapshot(state.q,state.v);feet=np.flatnonzero(state.contacts)
        jac=snap['foot_jacobian'][feet].reshape(-1,24);jdot=(snap['foot_jacobian_dot'][feet]@state.v).reshape(-1)
        stages=condense_implicit_drive(snap['mass_matrix'],snap['nonlinear'],S,interface.kp,interface.kd,
            jac,jdot,state.v,state.joint_position,substeps=16)
        constraints=interval_constraints(stages,S,effort=interface.effort,velocity_limit=interface.velocity,
            joint_limits=interface.limits,friction=interface.friction)
        response=constraints['matrix']@target
        violation=np.maximum(np.maximum(constraints['low']-response,response-constraints['high']),0.)
        categories={}
        for label,error in zip(constraints['labels'],violation):
            kind=label[1].split('_')[0] if label[1].startswith(('friction_','normal_')) else label[1]
            categories[kind]=max(categories.get(kind,0.),float(error))
        predicted=predict_implicit_support(model,interface,state,target,substeps=16)
        final=stages[-1]
        verror=float(np.max(abs(final['velocity_map']@target+final['velocity_bias']-predicted['state'].v)))
        qerror=float(np.max(abs(final['joint_position_map']@target+final['joint_position_bias']-predicted['state'].joint_position)))
        unsaturated=categories['motor_torque']<=3e-4
        if unsaturated and (verror>1e-6 or qerror>1e-8):
            raise ValueError('Condensed map disagrees with independent unsaturated substep integration')
        rows.append(dict(case=case,tick=tick,constraint_rows=len(violation),
            interval_feasible=bool(violation.max()<=3e-4),unsaturated=unsaturated,
            maximum_violations=categories,independent_velocity_error=verror,independent_joint_position_error=qerror))
    paths += [a.run/f for f in ('job.json','INPUTS.json','native_command_inputs.json','landing_mpc_records.json')]
    result=dict(schema='condensed-implicit-drive-audit-v1',samples=len(rows),substeps=16,
        unsaturated_samples=sum(r['unsaturated'] for r in rows),
        interval_feasible_samples=sum(r['interval_feasible'] for r in rows),records=rows,
        scope='Existing commands on reconstructed recorded states; unsaturated affine model checked against independent substeps. No new optimized actions or native execution.',
        source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_sha256')}))


if __name__=='__main__':main()
