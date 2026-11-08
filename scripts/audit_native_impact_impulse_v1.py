"""Infer contact impulses from measured momentum changes at selected impacts.

Least-squares inference is diagnostic, not measured friction or a replacement
contact law. Motor impulse uses the prior model estimate; unmodeled impulses
and local-frame variation remain in the reported residual.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.predict_speed_limited_spheres_v1 import surface_jacobians
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','run','baseline','comparison','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    baseline=read(args.baseline);comparison=read(args.comparison)
    pins=dict(comparison['source_sha256'])
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Changed evidence '+p)
    plan=read(args.plan);source=read(plan['source'])
    _,_,_,_,context=source_context(source['configuration']['source']);initialize(*context)
    model=_WORKER['controller'].qp.model;interface=_WORKER['controller'].qp.interface
    meta=read(args.run/'foot_impact_observation.json')
    with np.load(args.run/'foot_impact_observation.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    old={(r['case'],r['physics_frame']):r for r in baseline['rows']}
    selected=[r for r in comparison['rows'] if r['completed'] and not r['original_step_exact']]
    rows=[];dt=meta['physics_dt_s']
    for event in selected:
        case=event['case'];frame=event['physics_frame'];lane=meta['lanes'].index(case)
        i=int(np.flatnonzero(data['physics_frame']==frame)[0]);r=old[case,frame]
        states=[]
        for j in [i,i+1]:
            sample={k:v[j,lane] for k,v in data.items() if k!='physics_frame'}
            states.append(state_from_sample(model,sample,meta['origins_world'][lane]))
        before,after=states;snap=model.snapshot(before.q,before.v)
        measured_force=np.asarray(r['actual_final_foot_force_n'])
        loaded=np.flatnonzero(measured_force[:,2]>5.)
        jac=surface_jacobians(model,before.q,_WORKER['spheres'])[loaded].reshape(-1,24)
        motor=np.zeros(24);motor[model.v_indices]=r['motor_torque_nm']
        momentum=snap['mass_matrix']@(after.v-before.v)+dt*(snap['nonlinear']-motor)
        impulse,residual,rank,singular=np.linalg.lstsq(jac.T,momentum,rcond=None)
        impulses=impulse.reshape(-1,3);fitted=jac.T@impulse
        ratio=np.linalg.norm(impulses[:,:2],axis=1)/np.maximum(impulses[:,2],1e-12)
        # These are velocity of the same material contact points evaluated
        # with the initial Jacobian, not differentiated foot-center positions.
        vin=(jac@before.v).reshape(-1,3);vout=(jac@after.v).reshape(-1,3)
        rows.append(dict(case=case,physics_frame=frame,loaded_feet=[model.feet[k] for k in loaded],
            inferred_impulse_ns=impulses,measured_normal_impulse_ns=measured_force[loaded,2]*dt,
            model_impulse_ns=np.asarray(r['model_foot_impulse_ns'])[loaded],
            inferred_tangent_over_normal=ratio,configured_friction=float(interface.friction),
            incoming_material_point_velocity=vin,outgoing_material_point_velocity=vout,
            generalized_residual_l2=float(np.linalg.norm(momentum-fitted)),
            generalized_change_l2=float(np.linalg.norm(momentum)),
            residual_vector=momentum-fitted,least_squares_rank=int(rank),
            actual_max_joint_speed_fraction=float(np.max(abs(after.v[model.v_indices])/interface.velocity)),
            actual_endpoint_nonfoot_gap_m=r['actual_final_nonfoot_gap_m']))
    for p in [args.plan,args.baseline,args.comparison,Path(__file__).resolve()]:
        pins[str(p.resolve())]=sha(p)
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Source changed during analysis '+p)
    output=dict(scope=__doc__,rows=rows,source_sha256=pins,model_promoted=False,
        new_native_execution=False,valid_expert_labels=False,
        limitations=['Force sensor provides normal-force evidence, not a tangential measurement.',
            'Inference assumes contact locations and the existing rigid-body mass model.',
            'Generalized residual mixes force and torque units; not a physical energy norm.',
            'Finite-step drive, frame change, joint-limit and nonfoot impulses may bias inference.'])
    tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n');tmp.replace(args.output)
    print(json.dumps(plain(rows)))


if __name__=='__main__':main()
