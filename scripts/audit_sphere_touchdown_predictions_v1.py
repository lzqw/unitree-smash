"""Audit foot-sphere plane crossings in recorded native one-step predictions.

Authored geometry is not contact-force evidence. No future contact information
is used to select actions; all classifications are retrospective diagnostics.
The existing controller and queued experiment sources remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.summarize_precontact_prediction_v1 import phase,stats


def sphere_gaps(model,q,spheres):
    positions,rotations=model.poses(q)
    return np.array([positions[model.names.index(s['body'])][2]+
        (rotations[model.names.index(s['body'])]@np.asarray(s['center_body_m']))[2]-s['radius_m']
        for s in spheres])


def crossing_mask(start_gap,end_gap,start_contacts,tolerance):
    start_gap,end_gap=np.asarray(start_gap),np.asarray(end_gap)
    start_contacts=np.asarray(start_contacts)
    if (start_gap.shape!=(4,) or end_gap.shape!=(4,) or start_contacts.shape!=(4,)
            or start_contacts.dtype!=bool or not np.isfinite(start_gap).all()
            or not np.isfinite(end_gap).all() or not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('Require four finite gaps, Boolean observations and nonnegative tolerance')
    return (~start_contacts)&(start_gap>tolerance)&(end_gap<=tolerance)


def spheres_from_asset(asset,feet):
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();found={}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:continue
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body or body.GetName() not in feet:continue
        if prim.GetTypeName()!='Sphere':raise ValueError('Foot collider is not a sphere')
        name=body.GetName()
        if name in found:raise ValueError('Multiple foot colliders require explicit union')
        transform=np.asarray(cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse())
        if not np.allclose(transform[:3,:3]@transform[:3,:3].T,np.eye(3),atol=1e-10):
            raise ValueError('Scaled sphere unsupported')
        found[name]=dict(body=name,path=str(prim.GetPath()),center_body_m=transform[3,:3].tolist(),
            radius_m=float(prim.GetAttribute('radius').Get()),
            contact_offset_authored=prim.GetAttribute('physxCollision:contactOffset').Get(),
            rest_offset_authored=prim.GetAttribute('physxCollision:restOffset').Get())
    if set(found)!=set(feet):raise ValueError('Missing native foot sphere')
    return [found[n] for n in feet]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','audit','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    try:import pxr.Usd
    except ImportError:
        if os.environ.get('QUADSMASH_SPHERE_AUDIT_BOOTSTRAP'):raise
        libs=next((Path(sys.prefix)/'lib/python3.11/site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
        env=dict(os.environ);env['PYTHONPATH']=str(libs)+os.pathsep+env.get('PYTHONPATH','')
        env['LD_LIBRARY_PATH']=os.pathsep.join([str(libs/'bin'),str(Path(sys.prefix)/'lib'),env.get('LD_LIBRARY_PATH','')])
        env['QUADSMASH_SPHERE_AUDIT_BOOTSTRAP']='1'
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    from quadsmash.rl.episode_trace_v4 import unpack_public
    from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
    from scripts.implicit_support_prediction_v1 import predict_implicit_support
    from scripts.whole_body_support_qp_v1 import PositionPDInterface
    from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    audit=json.loads(a.audit.read_text())
    if Path(audit['source']).resolve()!=a.run.resolve():raise ValueError('Mismatched source')
    for path,digest in audit['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed prediction input '+path)
    protocol_path=ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol=json.loads(protocol_path.read_text());asset=ROOT/protocol['asset']['path']
    if sha(asset)!=protocol['asset']['sha256']:raise ValueError('Changed original asset')
    model=LoadedPinocchioModelV1();spheres=spheres_from_asset(asset,model.feet)
    native=json.loads((a.run/'native_command_inputs.json').read_text())
    limits=np.asarray(native['joint_limits_rad']);default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(limits,np.r_[np.full(12,25.),np.full(6,80.)],
        np.r_[np.full(12,.5),np.full(6,4.)],np.r_[np.full(12,23.5),np.full(6,100.)],
        np.r_[np.full(12,30.),np.full(6,5.)],.65)
    logs=json.loads((a.run/'landing_mpc_records.json').read_text())
    logs={(r['case'],r['control_tick']):r for r in logs if r.get('episode_index')==0}
    history=RecordedNonfootContactHistory(a.run);cache={};rows=[]
    for r in audit['records']:
        if not r['applied'] or phase(history.first_frame[r['case']]*history.dt,r['tick']) not in (
                'before_observed_nonfoot','no_recorded_nonfoot_event'):continue
        case,tick=r['case'],r['tick'];record=logs[(case,tick)]
        if case not in cache:
            with np.load(a.run/'fixed_policy_traces'/f'case_{case:03d}.npz') as z:cache[case]=unpack_public(z['packed_public'])
        public=cache[case];state=model.state_from_public(public[tick],default)
        actual=model.state_from_public(public[tick+1],default);observed=state.contacts.copy()
        state.contacts=np.asarray(record['support_contacts'],bool)
        target=np.asarray(record['final_dispatched_joint_target_rad'])
        prediction=predict_implicit_support(model,interface,state,target,substeps=16)['state']
        error=float(np.max(abs(prediction.joint_position[:12]-actual.joint_position[:12])))
        if abs(error-r['models']['implicit_16']['leg_position_error_max_rad'])>1e-10:
            raise ValueError('Does not reproduce frozen prediction audit')
        start_gap=sphere_gaps(model,state.q,spheres);actual_gap=sphere_gaps(model,actual.q,spheres)
        predicted_gap=sphere_gaps(model,prediction.q,spheres)
        rows.append(dict(case=case,tick=tick,full_horizon_accepted=bool(record['accepted']),
            leg_error_rad=error,observed_contact_change=r['observed_contact_change'],
            observed_contacts=observed.tolist(),next_observed_contacts=actual.contacts.tolist(),
            prediction_support_contacts=state.contacts.tolist(),start_gap_m=start_gap.tolist(),
            actual_next_gap_m=actual_gap.tolist(),fixed_mode_predicted_gap_m=predicted_gap.tolist(),
            geometric_crossing_by_tolerance={str(t):crossing_mask(start_gap,actual_gap,observed,t).tolist()
                for t in [1e-6,1e-4,1e-3]}))
    groups={}
    for tolerance in ['1e-06','0.0001','0.001']:
        crosses=lambda r:any(r['geometric_crossing_by_tolerance'][tolerance])
        groups[tolerance]={}
        for label,select in [('all',lambda r:True),('sphere_crosses_plane',crosses),
                            ('no_sphere_crossing',lambda r:not crosses(r)),
                            ('crossing_without_flag_change',lambda r:crosses(r) and not r['observed_contact_change'])]:
            chosen=[r for r in rows if select(r)]
            groups[tolerance][label]=dict(samples=len(chosen),cases=sorted({r['case'] for r in chosen}),
                leg_error_rad=stats([r['leg_error_rad'] for r in chosen]))
    paths=[Path(__file__),a.audit,protocol_path,asset,model.path,
        ROOT/'scripts/loaded_pinocchio_model_v1.py',ROOT/'scripts/implicit_support_prediction_v1.py',
        ROOT/'scripts/recorded_nonfoot_contact_history_v1.py',ROOT/'scripts/summarize_precontact_prediction_v1.py',
        a.run/'contact_telemetry_metadata.json',a.run/'first_nonfoot_contact_details.json',
        a.run/'fixed_policy_traces/manifest.json',a.run/'asset_dependencies.json']
    dependencies=json.loads((a.run/'asset_dependencies.json').read_text())['resolved_layers']
    paths.extend(Path(x) for x in dependencies)
    result=dict(scope=__doc__,run=str(a.run.resolve()),native_new_execution=False,valid_expert_labels=False,physical_cause_proven=False,
        original_outcomes_relabelled=False,spheres=spheres,groups=groups,records=rows,
        source_sha256={str(f.resolve()):sha(f) for f in paths},
        limitations=['Control-step endpoint geometry only, not continuous collision events',
            'Authored sphere with center offset; PhysX contact/rest margins are not reconstructed',
            'No recorded nonfoot event does not exclude unmonitored collision',
            'Comparison groups have different states; this is not causal controller ablation'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(groups))


if __name__=='__main__':main()
