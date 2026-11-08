"""Compare sustained controllers from actual native reference-end states.

Rebuild controller memory by consuming the recorded native prefix, verifying
its commands against read-back targets. Continue only the model from the real
handoff state. This does not teleport native physics or certify recovery.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.pure_control_settled_stance_v14 import PureControlSettledStanceV14
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def state_from_record(model, row):
    q,v=np.asarray(row['q']).copy(),np.asarray(row['v']).copy()
    r=pin.Quaternion(q[3:7]).matrix()
    state=PublicState(q,v,r,q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),
        np.asarray(row['phase']['ground_contacts'],bool),0.)
    state.previous_command_target=np.asarray(row['previous']).copy()
    state.measured_force_world=np.asarray(row['ground_force']).copy()
    return state


def evaluate(configuration,nominal,case,records,mode,deadline):
    if mode == 'v15':
        from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
        cls = PureControlStanceFootReferenceV15
    elif mode == 'v12':
        cls = PureControlReferenceClockV12
    elif mode == 'v14':
        cls = PureControlSettledStanceV14
    else:
        raise ValueError('Unknown controller mode '+mode)
    c=cls(PositionPDInterface(**configuration['interface']),nominal,configuration['model_reference_library'])
    errors=[]
    for row in records[:-1]:
        if time.monotonic()>deadline:
            return dict(case=case,mode=mode,failure=dict(reason='compute_budget_prefix'))
        state=state_from_record(c.model,row)
        target,_=c.command(state,row['phase']['phase'],row['time_s'])
        error=float(np.max(abs(target-np.asarray(row['actually_applied']))))
        errors.append(error)
        if error>2e-6:
            return dict(case=case,mode=mode,failure=dict(reason='prefix_command_reconstruction',
                tick=row['tick'],maximum_command_error_rad=error),native_success=False)
    current=records[-1]
    state=state_from_record(c.model,current)
    phase=current['phase']
    manager=EventRecoveryManagerV1()
    manager.memory=RecoveryPhaseMemory(episode=0,last_time_s=current['time_s'],phase=phase['phase'],
        release_time_s=phase['release_time_s'],first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=phase['control_sample_stable_duration_s'],
        ground_contacts=np.asarray(phase['ground_contacts'],bool).copy())
    begin=current['tick'];trace=[];failure=None;stable=0.;longest=0.;force=state.measured_force_world.copy()
    for tick in range(begin,800):
        if time.monotonic()>deadline:
            failure=dict(tick=tick,reason='compute_budget');break
        time_s=float(np.float32(tick)*np.float32(.01))
        if tick>begin:
            r=state.root_rotation_world
            phase=manager.update(episode=0,time_s=time_s,had_racket_contact=True,released=True,
                racket_touching=False,separated_now=True,ground_normal_force_n=force[:,2],
                roll_pitch_rad=[np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))],
                root_velocity_world=r@state.v[:3],root_angular_velocity_world=r@state.v[3:6],
                joint_velocity=state.joint_velocity,nonfoot_contact_now=False)
            state.contacts=phase['ground_contacts'].copy();state.measured_force_world=force.copy()
        target,diag=c.command(state,phase['phase'],time_s)
        recorder=ForceRecorder(c.model)
        try:
            prediction=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                substeps=160,geometry=c.path_geometry,collision_response=recorder)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(tick=tick,reason='model_rejected',error=repr(error));break
        bad={n:v for n,v in prediction['diagnostics'].items() if
            (n=='nonfoot_ground_violation_m' and v>1e-6) or
            (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            failure=dict(tick=tick,reason='physical_path_violation',violations=bad,
                collision=prediction['first_ground_violation'],source=diag['source']);break
        state=prediction['state'];state.previous_command_target=target.copy();force=recorder.last_force
        stable=phase['control_sample_stable_duration_s'];longest=max(longest,stable)
        trace.append(dict(tick=tick,phase=phase['phase'],source=diag['source'],stable_s=stable,
            terminal_stance_active=diag.get('terminal_stance_active',False),
            terminal_candidate_conditions=diag.get('terminal_candidate_conditions',{}),
            contacts=phase['ground_contacts'],root_position=state.q[:3].copy(),
            root_velocity_world=state.root_rotation_world@state.v[:3]))
    return dict(case=case,mode=mode,native_handoff_tick=begin,steps=len(trace),required_steps=800-begin,
        reconstructed_prefix_steps=len(errors),maximum_prefix_command_error_rad=max(errors,default=0.),
        failure=failure,final_model_stable_s=stable,longest_model_stable_s=longest,
        trajectory=trace,native_success=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','configuration','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cases',type=int,nargs='+',required=True)
    p.add_argument('--modes',nargs='+',choices=['v12','v14','v15'],default=['v12','v14'])
    args=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if 'v15' in args.modes:
        # Include the optional controller in the immutable source manifest
        # before spawning workers; legacy default modes are unchanged.
        from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
    if args.output.exists() or read(args.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Fresh audit and completed native source required')
    cfg=read(args.configuration);nominal=read(args.run/'native_command_inputs.json')['default_joint_position_rad']
    with np.load(args.run/'pure_control_commands.npz') as z:a={k:z[k].copy() for k in z.files}
    diagnostics=[json.loads(line) for line in (args.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(diagnostics)!=len(a['lane']):raise ValueError('Unaligned diagnostics')
    records={}
    for lane in args.cases:
        ids=np.flatnonzero(a['lane']==lane)
        if np.any(np.diff(a['tick'][ids])!=1):raise ValueError('Missing native control boundary')
        active=[i for i in ids if diagnostics[int(i)]['diagnostic'].get('reference_active')]
        if not active:raise ValueError('Selected case lacks an actual reference')
        end=int(a['tick'][active[-1]])+1;selected=[i for i in ids if a['tick'][i]<=end]
        if int(a['tick'][selected[-1]])!=end:raise ValueError('No native handoff state')
        records[lane]=[]
        for j,i in enumerate(selected):
            i=int(i);row=diagnostics[i]
            if (row['tick'],row['lane'])!=(int(a['tick'][i]),lane):raise ValueError('Misaligned row')
            record={k:a[k][i].copy() for k in ['q','v','previous','ground_force']}
            record.update(phase=row['phase'],tick=int(a['tick'][i]),time_s=float(a['time_s'][i]))
            if j+1<len(selected):
                record['actually_applied']=a['previous'][int(selected[j+1])].copy()
            records[lane].append(record)
    initialize=PureControlReferenceClockV12(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    files=[Path(__file__),args.configuration,Path(cfg['model_reference_library'])]
    files += [args.run/n for n in ['job.json','native_command_inputs.json','pure_control_commands.npz','pure_control_diagnostics.jsonl']]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    args.output.mkdir(parents=True)
    def write(name,data):
        f=args.output/name;t=f.with_suffix('.tmp');t.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');t.replace(f)
    write('inputs.json',dict(source_sha256=pins,cases=args.cases,modes=args.modes,workers=6,max_seconds=1800,
        native_state_teleport=False,model_start='actual native end-of-reference with reconstructed controller memory'))
    started=time.monotonic();results=[]
    write('status.json',dict(state='RUNNING_ACTUAL_HANDOFF_MODEL',pid=os.getpid(),completed=0,total=len(args.modes)*len(args.cases)))
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,lane,records[lane],mode,started+1800.)
              for lane in args.cases for mode in args.modes]
        for future in as_completed(jobs):
            r=future.result();write('case_%03d_%s.json'%(r['case'],r['mode']),r)
            results.append({k:v for k,v in r.items() if k!='trajectory'})
            write('status.json',dict(state='RUNNING_ACTUAL_HANDOFF_MODEL',pid=os.getpid(),completed=len(results),total=len(jobs)))
    for f,h in pins.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed input '+f)
    output=dict(state='COMPLETED_ACTUAL_HANDOFF_MODEL',results=results,wall_seconds=time.monotonic()-started,
        native_success=False,valid_expert_labels=False)
    write('result.json',output);write('status.json',output);print(json.dumps(plain(output)),flush=True)


if __name__=='__main__':main()
