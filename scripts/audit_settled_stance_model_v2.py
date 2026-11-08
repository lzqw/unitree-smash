"""State-qualified terminal stance using native lifecycle and model forces.

Ground forces are derived from the model's final microstep impulse. They are
not native sensor measurements. Physics, commands and constraints are intact;
this tests whether earlier model success depends on a different phase rule.
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
from scripts.pure_control_settled_stance_v14 import PureControlSettledStanceV14
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1
from scripts.audit_joint_sequence_proposal_v1 import plain


class ForceRecorder:
    def __init__(self,model):
        self.response=HipMountCollisionResponseV1(model)
        self.last_force=None
    def step(self,*args,**kwargs):
        result=self.response.step(*args,**kwargs)
        self.last_force=result['impulse'].reshape(4,3).copy()/float(args[8])
        return result


def evaluate(cfg,nominal,index,start_tick,deadline):
    c=PureControlSettledStanceV14(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    source=c.library.records[index];state=c.library.state(source,-1)
    if state.contacts.any():raise ValueError('Initial measured force needed for a grounded source')
    force=np.zeros((4,3));manager=EventRecoveryManagerV1()
    rows=[];failure=None;stable=0.;longest=0.;required=800-start_tick
    checks=0
    for k in range(required):
        if time.monotonic()>deadline:failure=dict(step=k,reason='compute_budget');break
        r=state.root_rotation_world;linear=r@state.v[:3]
        angles=[np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))]
        # The initial source is already a causally confirmed release. The
        # model starts there; this is not a new native takeover decision.
        phase=manager.update(episode=0,time_s=float(np.float32(start_tick+k)*np.float32(.01)),
            had_racket_contact=True,released=True,racket_touching=False,separated_now=True,
            ground_normal_force_n=force[:,2],roll_pitch_rad=angles,root_velocity_world=linear,
            root_angular_velocity_world=r@state.v[3:6],joint_velocity=state.joint_velocity,
            nonfoot_contact_now=False)
        impulse_contacts=state.contacts.copy();state.contacts=phase['ground_contacts'].copy()
        state.measured_force_world=force.copy()
        target,diag=c.command(state,phase['phase'],float(np.float32(start_tick+k)*np.float32(.01)))
        recorder=ForceRecorder(c.model)
        try:
            pred=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                substeps=160,geometry=c.path_geometry,collision_response=recorder)
            # Confirm observation instrumentation changes neither mechanics
            # nor the command, once in flight and once on actual model support.
            if checks==0 or (checks==1 and state.contacts.any()):
                ordinary=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                    substeps=160,geometry=c.path_geometry)
                if not np.array_equal(ordinary['state'].q,pred['state'].q) or not np.array_equal(ordinary['state'].v,pred['state'].v):
                    raise ValueError('Force recording altered model motion')
                checks+=1
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(step=k,reason='model_rejected',error=repr(error));break
        bad={n:v for n,v in pred['diagnostics'].items() if
            (n=='nonfoot_ground_violation_m' and v>1e-6) or
            (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            failure=dict(step=k,reason='physical_path_violation',violations=bad,
                collision=pred['first_ground_violation'],command_source=diag['source']);break
        force=recorder.last_force
        if force is None or not np.isfinite(force).all():raise ValueError('Missing finite model forces')
        state=pred['state'];state.previous_command_target=target.copy()
        stable=phase['control_sample_stable_duration_s'];longest=max(longest,stable)
        rows.append(dict(step=k,phase=phase['phase'],command_source=diag['source'],
            contact_mask=phase['ground_contacts'],prior_impulse_contacts=impulse_contacts,
            phase_mask_differs=bool(np.any(phase['ground_contacts']!=impulse_contacts)),
            terminal_stance_active=diag['terminal_stance_active'],
            terminal_candidate_conditions=diag['terminal_candidate_conditions'],
            terminal_stance_captured_tick=diag['terminal_stance_captured_tick'],
            final_microstep_ground_force_n=force,roll_pitch_rad=angles,root_velocity_world=linear,
            omega=float(np.linalg.norm(state.v[3:6])),joint_speed=float(np.max(abs(state.joint_velocity))),
            sampled_stable_s=stable))
    return dict(source_case=source['source_case'],steps=len(rows),required_steps=required,failure=failure,
        final_model_stable_s=stable,longest_model_stable_s=longest,trajectory=rows,
        force_recording_exact_motion_checks=checks,
        contact_mask_difference_steps=sum(r['phase_mask_differs'] for r in rows),
        native_phase_code_used=True,forces_are_model_impulse_estimates=True,
        native_success=False,valid_expert_labels=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['configuration','fixture-run','bank','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cases',nargs='+',type=int,required=True)
    args=parser.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if args.output.exists():raise ValueError('Fresh diagnostic required')
    cfg=read(args.configuration);refs=read(cfg['model_reference_library'])['references']
    nominal=read(args.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    bank={r['case']:r for r in read(args.bank)['queries']}
    indices=[i for i,r in enumerate(refs) if r['source_case'] in args.cases]
    if len(indices)!=len(set(args.cases)):raise ValueError('Missing selected references')
    args.output.mkdir(parents=True)
    def write(name,data):
        p=args.output/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');t.replace(p)
    files=[Path(__file__),args.configuration,args.bank,Path(cfg['model_reference_library']),args.fixture_run/'native_command_inputs.json']
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write('inputs.json',dict(source_sha256=pins,cases=args.cases,workers=6,max_seconds=1800,
        physical_substeps=160,force_units='impulse / actual microstep seconds',training=False))
    started=time.monotonic();results=[]
    write('status.json',dict(state='RUNNING_NATIVE_PHASE_MODEL',pid=os.getpid(),completed=0,total=len(indices)))
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate,cfg,nominal,i,int(bank[refs[i]['source_case']]['control_tick']),started+1800.) for i in indices]
        for future in as_completed(jobs):
            r=future.result();write('case_%03d.json'%r['source_case'],r)
            results.append({k:v for k,v in r.items() if k!='trajectory'})
            write('status.json',dict(state='RUNNING_NATIVE_PHASE_MODEL',pid=os.getpid(),completed=len(results),total=len(indices)))
    for name,expected in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Changed input '+name)
    summary=dict(state='COMPLETED_NATIVE_PHASE_MODEL',results=results,wall_seconds=time.monotonic()-started,
        native_success=False,valid_expert_labels=False)
    write('result.json',summary);write('status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
