"""Paired complete model executions of a new preloading WBC reference.

Uses actual release states and unchanged phase/stability code. No native
teleport or new learning. Candidate results do not promote a controller or
replace the separately running native full-cohort experiments.
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.pure_control_touchdown_velocity_v21 import PureControlTouchdownVelocityV21
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1,RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def controller(cfg,nominal,mode):
    cls=PureControlStanceFootReferenceV15 if mode=='baseline' else PureControlTouchdownVelocityV21
    return cls(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])


def evaluate(cfg,nominal,key,record,mode,deadline):
    c=controller(cfg,nominal,mode);state=state_from_record(c.model,record)
    phase=record['phase'];begin=int(record['tick']);manager=EventRecoveryManagerV1()
    manager.memory=RecoveryPhaseMemory(episode=0,last_time_s=record['time_s'],phase=phase['phase'],
        release_time_s=phase['release_time_s'],first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=0.,ground_contacts=state.contacts.copy())
    rows=[];failure=None;active_steps=0;qp_rejections=0
    for tick in range(begin,800):
        if time.monotonic()>deadline:failure=dict(tick=tick,reason='compute_budget');break
        now=float(np.float32(tick)*np.float32(.01))
        target,diagnostic=c.command(state,phase['phase'],now)
        if tick==begin:
            if c.chosen is not None:raise ValueError('Paired cohort must contain uncovered release states')
            if mode=='baseline' and np.max(abs(target-np.asarray(record['native_first_command'])))>1e-6:
                raise ValueError('Baseline does not reproduce first native command')
        td=diagnostic.get('touchdown_reference',{})
        active_steps+=int(td.get('active',False));qp_rejections+=int(not diagnostic['solver_accepted'])
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
                collision=prediction['first_ground_violation']);break
        state=prediction['state'];state.previous_command_target=target.copy();force=recorder.last_force
        r=state.root_rotation_world
        phase=manager.update(episode=0,time_s=float(np.float32(tick+1)*np.float32(.01)),
            had_racket_contact=True,released=True,racket_touching=False,separated_now=True,
            ground_normal_force_n=force[:,2],roll_pitch_rad=[np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))],
            root_velocity_world=r@state.v[:3],root_angular_velocity_world=r@state.v[3:6],
            joint_velocity=state.joint_velocity,nonfoot_contact_now=False)
        state.contacts=phase['ground_contacts'].copy();state.measured_force_world=force.copy()
        rows.append(dict(tick=tick,q=state.q.copy(),v=state.v.copy(),target=target.copy(),
            ground_force=force.copy(),phase=phase,command_source=diagnostic['source'],
            solver_accepted=diagnostic['solver_accepted'],touchdown_reference=td))
    return dict(state_key=key,mode=mode,initial_actual_state=record,steps=len(rows),required_steps=800-begin,
        failure=failure,final_model_stable_s=phase['control_sample_stable_duration_s'],
        full_model_path=failure is None and len(rows)==800-begin,active_reference_steps=active_steps,
        qp_rejections=qp_rejections,trajectory=rows,native_success=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-search','configuration','run','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    if a.output.exists() or read(a.run/'job.json')['status']!='SUCCEEDED':raise ValueError('Fresh model experiment required')
    original=read(a.source_search/'inputs.json')
    for f,digest in original['source_sha256'].items():
        if sha(f)!=digest:raise ValueError('Changed actual-state evidence '+f)
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    if original['source_sha256'].get(str(a.configuration.resolve()))!=sha(a.configuration):
        raise ValueError('Original native interface required')
    with np.load(a.run/'pure_control_commands.npz') as z:
        arrays={k:z[k].copy() for k in ['lane','tick','q','v','previous','ground_force','final','time_s']}
    diagnostics=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    records=original['states']
    for key,record in records.items():
        ids=np.flatnonzero((arrays['lane']==record['case']) & (arrays['tick']==record['tick']))
        if len(ids)!=1:raise ValueError('Unique release observation required')
        i=int(ids[0])
        if any(not np.array_equal(arrays[k][i],record[k]) for k in ['q','v','previous','ground_force']):
            raise ValueError('State differs from native evidence')
        record['native_first_command']=arrays['final'][i]
    # Check the five verified entries before the paired experiment. This is
    # mapping preservation at these inputs, not a new native success claim.
    preserved=[]
    for case in [30,42,50,85,90]:
        i=int(np.flatnonzero(arrays['lane']==case)[0])
        record={k:arrays[k][i] for k in ['q','v','previous','ground_force']}
        record['phase']=diagnostics[i]['phase']
        commands=[]
        for mode in ['baseline','candidate']:
            c=controller(cfg,nominal,mode);state=state_from_record(c.model,record)
            target,diag=c.command(state,record['phase']['phase'],float(arrays['time_s'][i]))
            if c.chosen is None:raise ValueError('Preserved case lacks selected model trajectory')
            commands.append(target)
            if mode=='candidate' and diag['touchdown_reference']['active']:
                raise ValueError('Candidate changed a selected reference')
        difference=float(np.max(abs(commands[0]-commands[1])))
        if difference>1e-12:raise ValueError('Selected-reference entry changed')
        preserved.append(dict(case=case,maximum_command_difference_rad=difference))
    files=[Path(__file__),a.configuration,a.source_search/'inputs.json']
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins=dict(original['source_sha256']);pins.update({str(f.resolve()):sha(f) for f in files})
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(source_sha256=pins,states=records,preserved_entries=preserved,
        workers=4,max_seconds=1200,physical_substeps=160,training=False,native_teleport=False,
        physical_limits_changed=False,acceptance_changed=False))
    started=time.monotonic();results=[]
    write('status.json',dict(state='RUNNING_PAIRED_TOUCHDOWN_CONTROL',pid=os.getpid(),completed=0,total=2*len(records)))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,key,record,mode,started+1200)
              for key,record in records.items() for mode in ['baseline','candidate']]
        for future in as_completed(jobs):
            result=future.result();write(result['state_key'].replace(':','_')+'_'+result['mode']+'.json',result)
            results.append({k:v for k,v in result.items() if k not in ['trajectory','initial_actual_state']})
            write('status.json',dict(state='RUNNING_PAIRED_TOUCHDOWN_CONTROL',pid=os.getpid(),completed=len(results),total=len(jobs),results=results))
    for f,digest in pins.items():
        if sha(f)!=digest:raise ValueError('Changed model experiment input '+f)
    result=dict(state='COMPLETED_PAIRED_TOUCHDOWN_CONTROL',results=results,preserved_entries=preserved,
        wall_seconds=time.monotonic()-started,native_success=False,valid_expert_labels=False)
    write('result.json',result);write('status.json',result);print(json.dumps(plain(result)))


if __name__=='__main__':main()
