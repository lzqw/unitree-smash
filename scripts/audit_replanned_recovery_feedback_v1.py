"""Execute newly optimized recovery plans with feedback to the original end.

Starts the CPU model at an actual recorded native state, installs the plan as
an optimization result, and invokes the full18 command selector and support
feedback every interval. No native state teleport or expert label is produced.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1,RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(cfg,nominal,path,deadline):
    result=json.loads(Path(path).read_text());fine=result['fine'];record=result['initial_actual_state']
    c=PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    state=state_from_record(c.model,record);phase=record['phase'];begin=record['tick']
    c.chosen=dict(query=dict(q=record['q'],v=record['v'],contacts=phase['ground_contacts'],target=record['previous']),
        trajectory=fine['trajectory'],source_key=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        native_success=False)
    c.selection_done=True;c.selection_score=None
    manager=EventRecoveryManagerV1()
    manager.memory=RecoveryPhaseMemory(episode=0,last_time_s=record['time_s'],phase=phase['phase'],
        release_time_s=phase['release_time_s'],first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=phase['control_sample_stable_duration_s'],ground_contacts=state.contacts.copy())
    rows=[];failure=None;force=state.measured_force_world.copy();stable=0.;longest=0.
    for tick in range(begin,800):
        if time.monotonic()>deadline:failure=dict(tick=tick,reason='compute_budget');break
        now=float(np.float32(tick)*np.float32(.01))
        if tick>begin:
            r=state.root_rotation_world
            phase=manager.update(episode=0,time_s=now,had_racket_contact=True,released=True,
                racket_touching=False,separated_now=True,ground_normal_force_n=force[:,2],
                roll_pitch_rad=[np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))],
                root_velocity_world=r@state.v[:3],root_angular_velocity_world=r@state.v[3:6],
                joint_velocity=state.joint_velocity,nonfoot_contact_now=False)
            state.contacts=phase['ground_contacts'].copy();state.measured_force_world=force.copy()
        target,diagnostic=c.command(state,phase['phase'],now)
        recorder=ForceRecorder(c.model)
        try:
            prediction=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                substeps=160,geometry=c.path_geometry,collision_response=recorder)
        except (ValueError,np.linalg.LinAlgError) as e:
            failure=dict(tick=tick,reason='model_rejected',error=repr(e));break
        bad={n:v for n,v in prediction['diagnostics'].items() if
            (n=='nonfoot_ground_violation_m' and v>1e-6) or
            (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            failure=dict(tick=tick,reason='physical_path_violation',violations=bad,
                collision=prediction['first_ground_violation'],source=diagnostic['source']);break
        state=prediction['state'];state.previous_command_target=target.copy();force=recorder.last_force
        stable=min((tick-begin)*.01,phase['control_sample_stable_duration_s']);longest=max(longest,stable)
        rows.append(dict(tick=tick,phase=phase['phase'],source=diagnostic['source'],
            q=state.q.copy(),v=state.v.copy(),target=target.copy(),ground_force=force.copy(),
            terminal_stance_active=diagnostic['terminal_stance_active'],stable_s=stable,
            reference_active=diagnostic['reference_active'],selected_predictive_candidate=diagnostic['selected_predictive_candidate']))
    return dict(state_key=result['state_key'],steps=len(rows),required_steps=800-begin,failure=failure,
        final_model_stable_s=stable,longest_model_stable_s=longest,trajectory=rows,
        native_success=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['configuration','fixture-run','search','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if a.output.exists() or read(a.search/'result.json')['state']!='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
        raise ValueError('Fresh completed-search feedback audit required')
    cfg=read(a.configuration);nominal=read(a.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    search_inputs=read(a.search/'inputs.json')
    for f,h in search_inputs['source_sha256'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed search input '+f)
    selected=[];excluded=[]
    for f in sorted(a.search.glob('state_*.json')):
        d=read(f);fine=d['fine']
        if fine['model_path_complete'] and fine['failure'] is None and fine['substeps']==160:selected.append(f)
        else:excluded.append(dict(state_key=d['state_key'],reason='No complete fine-model sequence'))
    a.output.mkdir(parents=True)
    def write(name,value):
        f=a.output/name;t=f.with_suffix('.tmp');t.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');t.replace(f)
    # Instantiate before taking the transitive source manifest.
    c=PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    files=[Path(__file__),a.configuration,a.fixture_run/'native_command_inputs.json',a.search/'inputs.json',a.search/'result.json',Path(cfg['model_reference_library']),*selected]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    write('inputs.json',dict(source_sha256=pins,excluded=excluded,native_teleport=False,max_seconds=1800,physical_substeps=160))
    start=time.monotonic();results=[]
    write('status.json',dict(state='RUNNING_REPLANNED_FEEDBACK_MODEL',pid=os.getpid(),completed=0,total=len(selected)))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,str(f),start+1800.) for f in selected]
        for job in as_completed(jobs):
            r=job.result();write('state_'+r['state_key'].replace(':','_')+'.json',r)
            results.append({k:v for k,v in r.items() if k!='trajectory'})
            write('status.json',dict(state='RUNNING_REPLANNED_FEEDBACK_MODEL',pid=os.getpid(),completed=len(results),total=len(jobs)))
    for f,h in pins.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed feedback input '+f)
    result=dict(state='COMPLETED_REPLANNED_FEEDBACK_MODEL',results=results,excluded=excluded,
        wall_seconds=time.monotonic()-start,native_success=False,valid_expert_labels=False)
    write('result.json',result);write('status.json',result);print(json.dumps(plain(result)),flush=True)


if __name__=='__main__':main()
