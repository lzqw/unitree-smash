"""Small explicit state perturbations of full-episode model recovery candidates.

Native trials continue independently. This diagnostic uses predicted impulse
contacts for phase selection, not the native force sensor hysteresis. It is
neither a native success test nor a source of expert labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,multiprocessing as mp,os,sys,time
from pathlib import Path
import numpy as np
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_trajectory_recovery_v9 import PureControlTrajectoryRecoveryV9
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(configuration,nominal,source_index,mode,deadline,required_steps):
    c=PureControlTrajectoryRecoveryV9(PositionPDInterface(**configuration['interface']),nominal,
        configuration['model_reference_library'])
    source=c.library.records[source_index];state=c.library.state(source,-1)
    if mode.startswith('velocity_'):
        axis=0 if '_x_' in mode else 2
        delta=np.zeros(3);delta[axis]=.025*(1 if mode.endswith('plus') else -1)
        state.v[:3]+=state.root_rotation_world.T@delta
    elif mode.startswith('pitch_'):
        linear=state.root_rotation_world@state.v[:3];angular=state.root_rotation_world@state.v[3:6]
        tangent=np.zeros(24);tangent[4]=.005*(1 if mode.endswith('plus') else -1)
        state.q=pin.integrate(c.model.model,state.q,tangent)
        state.root_rotation_world=pin.Quaternion(state.q[3:7]).matrix();state.root_position_world=state.q[:3].copy()
        state.v[:3]=state.root_rotation_world.T@linear;state.v[3:6]=state.root_rotation_world.T@angular
    else:raise ValueError('Explicit perturbation required')
    initial_match,initial_score=c.library.match(state)
    if initial_match is None:raise ValueError('Perturbation outside declared local reference coverage')
    rows=[];failure=None;settled=0.;stable=0.;longest=0.
    for k in range(required_steps):
        if time.monotonic()>deadline:failure=dict(step=k,reason='compute_budget');break
        r=state.root_rotation_world;linear=r@state.v[:3]
        angles=np.array([np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))])
        settling=state.contacts.any() and np.max(abs(angles))<np.deg2rad(30) and abs(linear[2])<.5 and np.linalg.norm(state.v[3:6])<2.
        settled=settled+.01 if settling else 0.
        phase='AIR_PREPARE' if not state.contacts.any() else ('SUSTAIN' if settled>=.15 else 'BUFFER_LOAD')
        if mode=='planned_then_feedback' and k<60:
            target=np.asarray(source['trajectory'][k]['target'],float);diagnostic=dict(source='planned_command',solver_accepted=None)
        else:
            # All variants initialize the actual controller on the initial
            # state. Planned-command mode still advances controller memory.
            target,diagnostic=c.command(state,phase,100.+k*.01)
        if mode=='planned_then_feedback' and k<60:
            c.command(state,phase,100.+k*.01)
        try:
            prediction=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                geometry=c.path_geometry,substeps=16)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(step=k,reason='model_rejected',error=repr(error));break
        bad={n:v for n,v in prediction['diagnostics'].items() if
            (n=='nonfoot_ground_violation_m' and v>1e-6) or
            (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            failure=dict(step=k,reason='physical_path_violation',violations=bad,
                collision=prediction['first_ground_violation'],command_source=diagnostic['source']);break
        state=prediction['state'];state.previous_command_target=target.copy()
        r=state.root_rotation_world;linear=r@state.v[:3]
        angles=np.array([np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))])
        stable_now=(state.contacts.any() and np.max(abs(angles))<=np.deg2rad(15) and
            np.linalg.norm(linear[:2])<=.3 and abs(linear[2])<=.1 and
            np.linalg.norm(state.v[3:6])<=1. and np.max(abs(state.joint_velocity))<=1.)
        stable=stable+.01 if stable_now else 0.;longest=max(longest,stable)
        reference=c.library.state(source,min(k,59))
        rows.append(dict(step=k,command_source=diagnostic['source'],solver_accepted=diagnostic['solver_accepted'],
            phase=phase,position_reference_error_m=float(np.linalg.norm(state.q[:3]-reference.q[:3])) if k<60 else None,
            maximum_joint_reference_error_rad=float(np.max(abs(state.joint_position-reference.joint_position))) if k<60 else None,
            roll_pitch_rad=angles,root_velocity_world=linear,omega=float(np.linalg.norm(state.v[3:6])),
            joint_speed=float(np.max(abs(state.joint_velocity))),support_feet=int(state.contacts.sum()),
            sampled_stable_s=stable))
    return dict(source_case=source['source_case'],source_key=source['source_key'],mode=mode,steps=len(rows),
        initial_match_score=initial_score,matched_reference_source=initial_match['source_key'],failure=failure,trajectory=rows,longest_model_stable_s=longest,final_model_stable_s=stable,required_steps=required_steps,model_only=True,
        phase_uses_predicted_impulse_contacts=True,full_episode_certified=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['configuration','fixture-run','bank','coarse-results','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=900.)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists() or not 60<=a.max_seconds<=1800:raise ValueError('Fresh bounded diagnostic required')
    config=read(a.configuration);nominal=read(a.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    library=read(config['model_reference_library']);eligible={r['source_case'] for f in a.coarse_results.glob('case_*.json') for r in [read(f)] if r['steps']==r['required_steps'] and r['final_model_stable_s']>=.8};indices=[i for i,r in enumerate(library['references']) if r['source_case'] in eligible];bank={r['case']:r for r in read(a.bank)['queries']};a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    pins={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),a.configuration,a.bank,a.fixture_run/'native_command_inputs.json',Path(config['model_reference_library'])]}
    write('inputs.json',dict(source_sha256=pins,max_seconds=a.max_seconds,workers=6,plant_substeps=16,controller_prediction_substeps=16,training=False))
    start=time.monotonic();deadline=start+a.max_seconds;results=[];total=6*len(indices)
    write('status.json',dict(state='RUNNING_MODEL_REFERENCE_TRACKING',pid=os.getpid(),completed=0,total=total))
    try:
        with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(evaluate,config,nominal,i,mode,deadline,800-int(bank[library['references'][i]['source_case']]['control_tick'])) for i in indices
                for mode in ['velocity_x_plus','velocity_x_minus','velocity_z_plus','velocity_z_minus','pitch_plus','pitch_minus']]
            for f in as_completed(futures):
                result=f.result();results.append(result)
                write('case_%03d_%s.json'%(result['source_case'],result['mode']),result)
                write('status.json',dict(state='RUNNING_MODEL_REFERENCE_TRACKING',pid=os.getpid(),completed=len(results),total=total))
        summaries={mode:dict(cases=sum(r['mode']==mode for r in results),
            model_600ms=sum(r['mode']==mode and r['steps']>=60 for r in results),
            model_1200ms=sum(r['mode']==mode and r['steps']>=120 for r in results),
            model_episode_end=sum(r['mode']==mode and r['steps']==r['required_steps'] for r in results),
            final_model_stable_08s=sum(r['mode']==mode and r['steps']==r['required_steps'] and r['final_model_stable_s']>=.8 for r in results)) for mode in ['velocity_x_plus','velocity_x_minus','velocity_z_plus','velocity_z_minus','pitch_plus','pitch_minus']}
        summary=dict(state='COMPLETED_MODEL_REFERENCE_TRACKING',pid=os.getpid(),results=summaries,
            wall_seconds=time.monotonic()-start,native_success=False,valid_expert_labels=False)
        write('result.json',summary);write('status.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as error:write('status.json',dict(state='FAILED',pid=os.getpid(),error=repr(error)));raise


if __name__=='__main__':main()
