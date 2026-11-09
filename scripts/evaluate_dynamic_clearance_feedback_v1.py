"""Full model feedback for contact-dynamic surface margins, from actual releases.

Independent experimental candidate; existing native inputs remain immutable.
Uses the same physical propagation and lifecycle as the paired settling runner.
Its completed archived V15 traces are the baseline; no new native results.
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
from scripts.pure_control_dynamic_clearance_v26 import PureControlDynamicClearanceV26
from scripts.pure_control_clearance_reference_v25 import surface_margin_penalty
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1, RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.retry_budgeted_settling_feedback_v1 import sha, write

def evaluate(cfg, nominal, path, mode, deadline):
    data = json.loads(Path(path).read_text())
    record = data['initial_actual_state']
    cls = PureControlDynamicClearanceV26 if mode == 'clearance' else PureControlStanceFootReferenceV15
    c = cls(PositionPDInterface(**cfg['interface']), nominal, cfg['model_reference_library'])
    state = state_from_record(c.model, record)
    phase = record['phase']
    c.chosen = dict(query=dict(q=record['q'], v=record['v'],
        contacts=phase['ground_contacts'], target=record['previous']),
        trajectory=data['fine']['trajectory'],
        source_key=hashlib.sha256(Path(path).read_bytes()).hexdigest(), native_success=False)
    c.selection_done = True
    c.selection_score = None
    manager = EventRecoveryManagerV1()
    manager.memory = RecoveryPhaseMemory(episode=0, last_time_s=record['time_s'],
        phase=phase['phase'], release_time_s=phase['release_time_s'],
        first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=0., ground_contacts=state.contacts.copy())
    rows = []
    failure = None
    capture_tick = None
    for tick in range(record['tick'], 800):
        if time.monotonic() > deadline:
            failure = dict(tick=tick, reason='compute_budget')
            break
        now = float(np.float32(tick)*np.float32(.01))
        target, diag = c.command(state, phase['phase'], now)
        if diag.get('state_settling_capture_now'):
            capture_tick = tick
        recorder = ForceRecorder(c.model)
        try:
            pred = predict_self_contact_spheres(c.model, c.interface, state, target,
                c.geometry.spheres, substeps=160, geometry=c.path_geometry,
                collision_response=recorder)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(tick=tick, reason='model_rejected', error=repr(error))
            break
        bad = {n:v for n,v in pred['diagnostics'].items() if
            (n == 'nonfoot_ground_violation_m' and v > 1e-6) or
            (n in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                'motor_torque_violation_nm'] and v > 3e-4)}
        if bad:
            failure = dict(tick=tick, reason='physical_path_violation',
                violations=bad, collision=pred['first_ground_violation'])
            break
        state = pred['state']
        state.previous_command_target = target.copy()
        force = recorder.last_force
        r = state.root_rotation_world
        phase = manager.update(episode=0,
            time_s=float(np.float32(tick+1)*np.float32(.01)),
            had_racket_contact=True, released=True, racket_touching=False, separated_now=True,
            ground_normal_force_n=force[:, 2],
            roll_pitch_rad=[np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(-r[2, 0], -1, 1))],
            root_velocity_world=r@state.v[:3], root_angular_velocity_world=r@state.v[3:6],
            joint_velocity=state.joint_velocity, nonfoot_contact_now=False)
        state.contacts = phase['ground_contacts'].copy()
        state.measured_force_world = force.copy()
        rows.append(dict(tick=tick, q=state.q.copy(), v=state.v.copy(), target=target.copy(),
            ground_force=force.copy(), phase=phase, source=diag['source'],
            solver_accepted=diag['solver_accepted'],
            terminal_stance_active=diag['terminal_stance_active'],
            reference_active=diag['reference_active']))
    return dict(state_key=data['state_key'], mode=mode, steps=len(rows),
        required_steps=800-record['tick'], failure=failure,
        full_model_path=failure is None and len(rows)==800-record['tick'],
        final_model_stable_s=phase['control_sample_stable_duration_s'],
        state_settling_capture_tick=capture_tick, trajectory=rows,
        initial_actual_state=record, native_success=False, valid_expert_labels=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['feedback','configuration','run','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--states',nargs='+',required=True)
    a=parser.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh experiment output required')
    inputs=read(a.feedback/'inputs.json');result=read(a.feedback/'result.json')
    if result['state']!='COMPLETED_ARCHIVED_REFERENCE_RECHECK' or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Completed actual-state baseline required')
    pins=dict(inputs['source_sha256'])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed baseline '+f)
    if pins.get(str(a.configuration.resolve()))!=sha(a.configuration):
        raise ValueError('Keep baseline execution configuration')
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    paths=[];baselines=[];zero_checks=[]
    with np.load(a.run/'pure_control_commands.npz') as z:
        for key in a.states:
            name='state_'+key.replace(':','_')+'.json'
            path=a.feedback/'candidate_snapshots'/name;data=read(path)
            record=data['initial_actual_state'];baseline_path=a.feedback/name;baseline=read(baseline_path)
            if baseline['failure'] is not None or not baseline['full_model_path'] or baseline['native_success']:
                raise ValueError('Completed model-only baseline required')
            if baseline['initial_actual_state']!=record or data['state_key']!=key or data['fine']['substeps']!=160:
                raise ValueError('Mismatched baseline initial state or physical precision')
            idx=np.flatnonzero((z['lane']==record['case'])&(z['tick']==record['tick']))
            if len(idx)!=1 or any(not np.array_equal(z[k][idx[0]],record[k]) for k in ['q','v','previous','ground_force']):
                raise ValueError('Not rooted at the actual native release')
            commands=[]
            for cls,options in [(PureControlStanceFootReferenceV15,{}),(PureControlDynamicClearanceV26,dict(clearance_weight=0.))]:
                c=cls(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'],**options)
                c.chosen=dict(query=dict(q=record['q'],v=record['v'],contacts=record['phase']['ground_contacts'],target=record['previous']),trajectory=data['fine']['trajectory'],source_key=sha(path),native_success=False)
                c.selection_done=True;c.selection_score=None
                target,_=c.command(state_from_record(c.model,record),record['phase']['phase'],record['time_s'])
                commands.append(target)
            error=float(np.max(abs(commands[0]-commands[1])))
            if error>1e-12:raise ValueError('Zero-weight entry differs from original controller')
            zero_checks.append(dict(state_key=key,maximum_request_error_rad=error))
            paths.append(path);baselines.append(baseline_path)
    # Verify the cost's physical sign and separation of ground and obstacles.
    def cost(gap,rate):return surface_margin_penalty({'Head_lower':dict(gap_m=gap,rate_m_s=rate),'FL_foot':dict(gap_m=-1,rate_m_s=-1),'Racket::net':dict(gap_m=-1,rate_m_s=-1)},approach_s=0.)[0]
    if not (cost(.02,0)==0 and cost(.005,-.1)==cost(.005,0)>cost(.009,0) and cost(.005,.1)==cost(.005,0)):
        raise ValueError('Invalid margin/rate sign or obstacle scope')
    a.output.mkdir(parents=True)
    files=[Path(__file__),a.configuration,a.feedback/'inputs.json',a.feedback/'result.json',a.run/'job.json',a.run/'pure_control_commands.npz',a.run/'native_command_inputs.json',*paths,*baselines]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins.update({str(f.resolve()):sha(f) for f in files})
    write(a.output/'inputs.json',dict(source_sha256=pins,states=a.states,workers=3,max_seconds=2400,physical_substeps=160,scoring_substeps=16,zero_weight_entry_checks=zero_checks,cost_semantics_checked=True,soft_margin_m=.01,approach_s=0.,dynamic_prediction_s=.05,weight=10.,native_execution=False,training=False,acceptance_changed=False,physical_interface_unchanged=True,baseline_feedback=[str(p.resolve()) for p in baselines]))
    start=time.monotonic();rows=[]
    def progress(state):write(a.output/'status.json',dict(state=state,pid=os.getpid(),completed=len(rows),total=len(paths),results=rows,elapsed_s=time.monotonic()-start,native_success=False))
    progress('RUNNING_DYNAMIC_CLEARANCE_FEEDBACK')
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,str(path),'clearance',start+2400) for path in paths]
        for job in as_completed(jobs):
            row=job.result();write(a.output/('state_'+row['state_key'].replace(':','_')+'.json'),row)
            rows.append({k:v for k,v in row.items() if k not in ['trajectory','initial_actual_state']})
            progress('RUNNING_DYNAMIC_CLEARANCE_FEEDBACK')
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed experiment input '+f)
    summary=dict(state='COMPLETED_DYNAMIC_CLEARANCE_FEEDBACK',results=rows,elapsed_s=time.monotonic()-start,native_execution=False,native_success=False,valid_expert_labels=False)
    write(a.output/'result.json',summary);progress(summary['state']);print(json.dumps(summary))

if __name__=='__main__':main()
