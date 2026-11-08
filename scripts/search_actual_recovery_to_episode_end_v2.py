"""Full18 sequence search through the actual episode end from a measured state.

This is the offline planning part of a prospective receding-horizon controller,
not a native restart or a deployed MPC. Every command knot, including the
first, is adjustable within original joint ranges. Future native actions are
not used to initialize, score or select candidates.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_reference_clock_v12 import PureControlReferenceClockV12
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_settled_stance_model_v2 import ForceRecorder
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1,RecoveryPhaseMemory
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain

KNOTS=np.array([0.,.10,.25,.50,.80,1.20,1.60,2.00,2.50])
STEPS=250
WORK={}


def initialize(configuration,nominal,records):
    WORK.update(c=PureControlReferenceClockV12(PositionPDInterface(**configuration['interface']),nominal,
        configuration['model_reference_library']),records=records,nominal=np.asarray(nominal))


def rollout(key,nodes,substeps,deadline):
    c=WORK['c'];record=WORK['records'][key];state=state_from_record(c.model,record)
    phase=record['phase'];manager=EventRecoveryManagerV1()
    manager.memory=RecoveryPhaseMemory(episode=0,last_time_s=record['time_s'],phase=phase['phase'],
        release_time_s=phase['release_time_s'],first_ground_time_s=phase['first_ground_time_s'],
        settled_duration_s=phase['phase_settled_duration_s'],
        final_stable_duration_s=phase['control_sample_stable_duration_s'],ground_contacts=state.contacts.copy())
    limits=c.interface.limits;span=np.diff(limits).ravel()
    targets=limits[:,0]+.5*(np.clip(nodes,-1,1)+1)*span
    curve=PchipInterpolator(KNOTS,targets,axis=0)
    previous=state.previous_command_target.copy();rows=[];cost=0.;failure=None;stable=0.;longest=0.
    for k in range(STEPS):
        if time.monotonic()>deadline:failure=dict(step=k,reason='compute_budget');break
        target=np.clip(curve(k*.01),limits[:,0],limits[:,1])
        recorder=ForceRecorder(c.model)
        try:
            pred=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                substeps=substeps,geometry=c.path_geometry,collision_response=recorder)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(step=k,reason='model_rejected',error=repr(error));break
        bad={n:v for n,v in pred['diagnostics'].items() if
            (n=='nonfoot_ground_violation_m' and v>1e-6) or
            (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:
            failure=dict(step=k,reason='physical_path_violation',violations=bad,collision=pred['first_ground_violation']);break
        state=pred['state'];state.previous_command_target=target.copy();force=recorder.last_force
        r=state.root_rotation_world;linear=r@state.v[:3]
        angles=np.array([np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))])
        phase=manager.update(episode=0,time_s=float(np.float32(record['tick']+k+1)*np.float32(.01)),
            had_racket_contact=True,released=True,racket_touching=False,separated_now=True,
            ground_normal_force_n=force[:,2],roll_pitch_rad=angles,root_velocity_world=linear,
            root_angular_velocity_world=r@state.v[3:6],joint_velocity=state.joint_velocity,nonfoot_contact_now=False)
        state.contacts=phase['ground_contacts'].copy();state.measured_force_world=force.copy()
        # Only stability accumulated during this newly planned segment counts
        # toward its terminal window; no earlier native window is credited.
        stable=min((k+1)*.01,phase['control_sample_stable_duration_s']);longest=max(longest,stable)
        omega=float(np.linalg.norm(state.v[3:6]));joint_speed=float(np.max(abs(state.joint_velocity)))
        cost+=.01*(20.*np.sum(angles**2)+10.*np.sum(linear**2)+5.*omega**2+.05*joint_speed**2)
        cost+=.001*np.sum(((target-previous)/span)**2);previous=target.copy()
        rows.append(dict(step=k,q=state.q.copy(),v=state.v.copy(),target=target.copy(),
            contacts=state.contacts.copy(),ground_force=force.copy(),phase=phase,roll_pitch=angles,
            root_velocity_world=linear,omega=omega,joint_speed=joint_speed,stable_s=stable))
    full=failure is None and len(rows)==STEPS
    if rows:
        last=rows[-1];cost+=30.*np.sum(last['roll_pitch']**2)+30.*np.sum(last['root_velocity_world']**2)+10.*last['omega']**2+.2*last['joint_speed']**2
    return dict(nodes=nodes,steps=len(rows),required_steps=STEPS,model_path_complete=full,failure=failure,
        terminal_stable_s=stable,longest_stable_s=longest,cost=float(cost),
        rank=[0 if full else 1,STEPS-len(rows),0 if stable>=.8 else 1,-stable,float(cost)],
        terminal=rows[-1] if rows else None,trajectory=rows if substeps==160 else None,
        substeps=substeps,native_success=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','configuration','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--states',nargs='+',required=True,help='Actual states exactly 250 ticks before episode end')
    p.add_argument('--warmstart',type=Path,required=True,help='Earlier model-only planned sequence from the exact same state')
    p.add_argument('--max-seconds',type=float,default=1800.)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if a.output.exists() or not 900<=a.max_seconds<=3600 or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Fresh bounded search from completed native evidence required')
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    with np.load(a.run/'pure_control_commands.npz') as z:arrays={k:z[k].copy() for k in z.files}
    diagnostics=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    with np.load(a.run/'pure_control_ground_forces.npz') as z:peaks=z['interval_peak_normal_n'].copy()
    names=read(a.run/'pure_control_ground_forces.json')['body_names'];nonfoot=[i for i,n in enumerate(names) if not n.endswith('_foot')]
    records={}
    for key in a.states:
        case,tick=map(int,key.split(':'));ids=np.flatnonzero((arrays['lane']==case)&(arrays['tick']==tick))
        if len(ids)!=1 or tick+STEPS!=800:raise ValueError('Require full remaining 2.5-second episode, without truncation')
        i=int(ids[0]);row=diagnostics[i]
        if (row['lane'],row['tick'])!=(case,tick) or not row['phase']['owns_final_recovery_command']:
            raise ValueError('State must already belong to causal post-release control')
        start=int(arrays['tick'][np.flatnonzero(arrays['lane']==case)[0]])
        if np.any(peaks[start:tick,case][:,nonfoot]>0.):raise ValueError('Selected state already has landing-segment nonfoot contact')
        record={k:arrays[k][i].copy() for k in ['q','v','previous','ground_force']}
        record.update(case=case,tick=tick,time_s=float(arrays['time_s'][i]),phase=row['phase'])
        records[key]=record
    warm=read(a.warmstart)
    key=warm['state_key']
    if key not in records:raise ValueError('Warmstart state not selected')
    for field in ['q','v','previous','ground_force']:
        if not np.array_equal(np.asarray(warm['initial_actual_state'][field]),records[key][field]):
            raise ValueError('Warmstart must come from identical measured state: '+field)
    if warm['native_success'] or warm['valid_expert_labels']:raise ValueError('Warmstart must remain a model-only candidate')
    old_nodes=np.asarray(warm['fine']['nodes'])
    if old_nodes.shape!=(8,18):raise ValueError('Unexpected prior sequence')
    seed_nodes=np.concatenate([old_nodes,old_nodes[-1:]],axis=0)
    initialize(cfg,nominal,records);c=WORK['c'];limits=c.interface.limits;span=np.diff(limits).ravel()
    normalize=lambda target:np.clip(2*(np.asarray(target)-limits[:,0])/span-1,-1,1)
    a.output.mkdir(parents=True)
    def write(name,data):
        f=a.output/name;t=f.with_suffix('.tmp');t.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');t.replace(f)
    files=[Path(__file__),a.configuration,a.warmstart,Path(cfg['model_reference_library'])]
    files += [a.run/n for n in ['job.json','native_command_inputs.json','pure_control_commands.npz','pure_control_diagnostics.jsonl','pure_control_ground_forces.json','pure_control_ground_forces.npz']]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    write('inputs.json',dict(source_sha256=pins,states=records,max_seconds=a.max_seconds,workers=8,
        knots_s=KNOTS,dimensions=18,first_command_free=True,future_native_actions_used=False,
        native_state_teleport=False,training=False,native_success=False,
        plans_through_original_episode_end=True,warmstart_model_only=True))
    started=time.monotonic();deadline=started+a.max_seconds
    candidates={key:[] for key in records};rng={key:np.random.default_rng(886000+i) for i,key in enumerate(records)}
    completed=0
    write('status.json',dict(state='SEARCHING_ACTUAL_RECOVERY_SEQUENCES',pid=os.getpid(),completed_candidates=0))
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cfg,nominal,records)) as pool:
        jobs=[(warm['state_key'],pool.submit(rollout,warm['state_key'],seed_nodes,16,deadline-600.))]
        for key,record in records.items():
            q=record['q'][c.model.q_indices];previous=record['previous']
            goals=[q,previous,np.asarray(nominal),.5*(q+nominal)]
            for goal in goals:
                for transition in [.25,1.0]:
                    fraction=np.minimum(KNOTS/transition,1.)[:,None]
                    nodes=normalize(previous[None,:]+fraction*(goal-previous)[None,:])
                    jobs.append((key,pool.submit(rollout,key,nodes,16,deadline-600.)))
        for key,f in jobs:candidates[key].append(f.result());completed+=1
        for generation,sigma in enumerate([.35,.25,.16,.10,.06]):
            if time.monotonic()>=deadline-600.:break
            jobs=[]
            for key in records:
                elite=sorted(candidates[key],key=lambda r:r['rank'])[:4]
                for j in range(24):
                    physical=rng[key].normal(0,.3*sigma,(len(KNOTS),18))
                    extension=rng[key].normal(0,sigma,(len(KNOTS),1))+rng[key].normal(0,.4*sigma,(len(KNOTS),4))
                    physical[:,8:12]+=extension;physical[:,4:8]-=.5*extension
                    physical[:,:4]+=rng[key].normal(0,.5*sigma,(len(KNOTS),1))
                    physical[:,12:]+=rng[key].normal(0,sigma,(len(KNOTS),6))
                    physical[1:]=.5*physical[1:]+.5*physical[:-1]
                    nodes=np.clip(np.asarray(elite[j%len(elite)]['nodes'])+2*physical/span,-1,1)
                    jobs.append((key,pool.submit(rollout,key,nodes,16,deadline-600.)))
            for key,f in jobs:candidates[key].append(f.result());completed+=1
            write('status.json',dict(state='SEARCHING_ACTUAL_RECOVERY_SEQUENCES',pid=os.getpid(),generation=generation+1,
                completed_candidates=completed,elapsed_s=time.monotonic()-started))
        fine={key:[] for key in records};jobs=[]
        for key in records:
            for rank,row in enumerate(sorted(candidates[key],key=lambda r:r['rank'])[:3]):
                jobs.append((key,rank,pool.submit(rollout,key,np.asarray(row['nodes']),160,deadline)))
        for key,rank,f in jobs:fine[key].append(dict(coarse_rank=rank,result=f.result()))
    summary=[]
    for key in records:
        chosen=min(fine[key],key=lambda r:r['result']['rank'])['result']
        result=dict(state_key=key,initial_actual_state=records[key],fine=chosen,
            coarse_candidates=[{k:v for k,v in r.items() if k not in ['trajectory','terminal']} for r in sorted(candidates[key],key=lambda r:r['rank'])],
            fine_alternatives=[{k:v for k,v in r['result'].items() if k not in ['trajectory','terminal']} for r in fine[key]],
            model_horizon_reaches_original_episode_end=True,full_episode_certified=False,native_success=False,valid_expert_labels=False)
        write('state_'+key.replace(':','_')+'.json',result)
        summary.append(dict(state_key=key,steps=chosen['steps'],model_path_complete=chosen['model_path_complete'],
            terminal_stable_s=chosen['terminal_stable_s'],failure=chosen['failure']))
    for f,h in pins.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed search source '+f)
    result=dict(state='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH',completed_candidates=completed,results=summary,
        wall_seconds=time.monotonic()-started,native_success=False,valid_expert_labels=False)
    write('result.json',result);write('status.json',result);print(json.dumps(plain(result)),flush=True)


if __name__=='__main__':main()
