"""Measure one-step model errors using actual state and actual held command.

Each prediction starts from an independently recorded native control boundary.
This removes accumulated nominal-reference drift from the comparison. It does
not reset native physics, fit parameters, or certify a controller.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain

_WORK = {}


def check_alignment(arrays, records):
    n = len(arrays['q'])
    if n < 2 or len(records) != n:
        raise ValueError('At least two matched actual control boundaries required')
    ticks = [r['control_tick'] for r in records]
    if ticks != list(range(ticks[0], ticks[0]+n)):
        raise ValueError('Contiguous physical control boundaries required')
    for key, width in [('q',25), ('v',24), ('previous_target_rad',18), ('final_target_rad',18)]:
        if arrays[key].shape != (n,width) or not np.isfinite(arrays[key]).all():
            raise ValueError('Malformed native input '+key)
    if not np.array_equal(arrays['previous_target_rad'][1:], arrays['final_target_rad'][:-1]):
        raise ValueError('Next boundary history does not match actual dispatched command')
    return ticks


def initialize(configuration, spheres, arrays):
    controller = build_controller(configuration)
    model = controller.qp.model
    _WORK.update(controller=controller, model=model, arrays=arrays, spheres=spheres,
        geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json', model, margin_m=0.))


def predict_step(index, tick, deadline):
    if time.monotonic() >= deadline:
        return dict(tick=tick, completed=False, reason='declared_compute_budget')
    w = _WORK; a = w['arrays']; model = w['model']
    q, v = a['q'][index].copy(), a['v'][index].copy()
    rotation = pin.Quaternion(q[3:7]).matrix()
    state = PublicState(q,v,rotation,q[:3].copy(),q[model.q_indices].copy(),
        v[model.v_indices].copy(),a['contacts'][index].copy(),0.)
    state.previous_command_target = a['previous_target_rad'][index].copy()
    target = a['final_target_rad'][index].copy()
    try:
        prediction = predict_frictional_spheres(model,w['controller'].qp.interface,state,target,
            w['spheres'],geometry=w['geometry'],substeps=160)
    except (ValueError,np.linalg.LinAlgError) as error:
        return dict(tick=tick,completed=False,reason='model_solver_rejected',error=repr(error))
    predicted = prediction['state']; actual_q,actual_v = a['q'][index+1],a['v'][index+1]
    actual_rotation = pin.Quaternion(actual_q[3:7]).matrix()
    tangent = pin.difference(model.model, predicted.q, actual_q)
    joint_error = predicted.joint_position-actual_q[model.q_indices]
    velocity_error = predicted.joint_velocity-actual_v[model.v_indices]
    worst = int(np.argmax(abs(velocity_error)))
    return dict(tick=tick,completed=True,root_position_error_m=float(np.linalg.norm(predicted.q[:3]-actual_q[:3])),
        root_attitude_error_rad=float(np.linalg.norm(tangent[3:6])),
        root_linear_velocity_error_m_s=float(np.linalg.norm(predicted.root_rotation_world@predicted.v[:3]-actual_rotation@actual_v[:3])),
        root_angular_velocity_error_rad_s=float(np.linalg.norm(predicted.root_rotation_world@predicted.v[3:6]-actual_rotation@actual_v[3:6])),
        joint_position_error_max_rad=float(np.max(abs(joint_error))),
        joint_velocity_error_max_rad_s=float(np.max(abs(velocity_error))),
        worst_velocity_joint=model.joint_order[worst], joint_velocity_error_rad_s=velocity_error,
        joint_position_error_rad=joint_error,initial_contacts=a['contacts'][index],
        predicted_loaded_feet=predicted.contacts, next_native_contact_observation=a['contacts'][index+1],
        contact_sets_equal=bool(np.array_equal(predicted.contacts,a['contacts'][index+1])),
        diagnostics=prediction['diagnostics'],first_ground_violation=prediction['first_ground_violation'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['run','bundle','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-seconds',type=float,default=600.)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=4 or not 60<=args.max_seconds<=900:
        raise ValueError('Fresh output and bounded computation required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=read(args.bundle/'manifest.json');result=read(args.run/'sequence_native_result.json')
    if result['case']!=3 or result['start_tick']!=280:
        raise ValueError('Declared TRAIN3 early-reference diagnostic required')
    if read(args.run/'job.json')['status']!='SUCCEEDED':raise ValueError('Completed native run required')
    with np.load(args.run/'sequence_native_trace.npz',allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in archive.files}
    ticks=check_alignment(arrays,result['records'])
    telemetry=read(args.run/'contact_telemetry_metadata.json')
    events=read(args.run/'first_nonfoot_contact_details.json')
    first_nonfoot_s=min((e['physics_frame']*telemetry['physics_dt_s'] for e in events
                        if 3 in e['first_event_env_ids']), default=None)
    event_tick=int(np.floor(first_nonfoot_s/.01)) if first_nonfoot_s is not None else 800
    sphere_paths=[Path(f) for f in manifest['source_sha256'] if Path(f).name=='sphere_crossings_v3.json']
    if len(sphere_paths)!=1:raise ValueError('Unique authored foot geometry required')
    paths=[args.bundle/'manifest.json',args.run/'sequence_native_trace.npz',args.run/'sequence_native_result.json',
        args.run/'job.json',args.run/'contact_telemetry_metadata.json',args.run/'first_nonfoot_contact_details.json',
        sphere_paths[0],Path(__file__),ROOT/'configs/go2_piper_authored_ground_geometry_v1.json']
    hashes=dict(manifest['source_sha256']);hashes.update({str(f.resolve()):sha(f) for f in paths})
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            f=Path(module.__file__).resolve();hashes[str(f)]=sha(f)
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed input '+f)
    started=time.monotonic();deadline=started+args.max_seconds;rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
        initializer=initialize,initargs=(manifest['configuration'],read(sphere_paths[0])['spheres'],arrays)) as pool:
        futures=[pool.submit(predict_step,i,t,deadline) for i,t in enumerate(ticks[:-1])]
        for future in as_completed(futures):rows.append(future.result())
    rows.sort(key=lambda r:r['tick'])
    keys=['root_position_error_m','root_attitude_error_rad','root_linear_velocity_error_m_s',
          'root_angular_velocity_error_rad_s','joint_position_error_max_rad','joint_velocity_error_max_rad_s']
    # Time slices describe different execution phases, not independent trials.
    groups=[]
    for name,start,end in [('all',280,800),('flight_and_first_landing',280,330),
        ('support_before_nonfoot_event',330,event_tick),('nonfoot_event_interval_and_after',event_tick,800)]:
        selected=[r for r in rows if start<=r['tick']<end and r['completed']]
        groups.append(dict(name=name,start_tick=start,end_tick_exclusive=end,samples=len(selected),
            errors={k:dict(median=float(np.median([r[k] for r in selected])),p95=float(np.percentile([r[k] for r in selected],95)),
                           maximum=float(max(r[k] for r in selected))) for k in keys} if selected else {},
            contact_set_mismatch=sum(not r['contact_sets_equal'] for r in selected)))
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Input changed during audit '+f)
    output=dict(scope=__doc__,run=str(args.run),first_original_nonfoot_contact_s=first_nonfoot_s,
        source_sha256=hashes,rows=rows,groups=groups,
        completed_intervals=sum(r['completed'] for r in rows),required_intervals=len(ticks)-1,
        wall_seconds=time.monotonic()-started,actual_commands_used=True,accumulated_prediction_error=False,
        native_execution=False,valid_expert_labels=False,
        limitations=['Replays recorded native transitions; does not execute new native controls.',
            'Model omits nonfoot impact response and self-collision; post-contact errors do not isolate PD accuracy.',
            'Loaded impulse support and native force-hysteresis observations are different contact definitions.',
            'Native endpoints are available only through tick799; the last command has no recorded next state.',
            'Time slices are diagnostic, not acceptance thresholds or independent generalization samples.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps(plain({k:output[k] for k in ['completed_intervals','required_intervals','wall_seconds','groups']})),flush=True)


if __name__=='__main__':main()
