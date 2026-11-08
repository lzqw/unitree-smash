"""Optimize longer full18 recovery tails for ball-qualified failure states.

Fixed first 300 ms of an existing model-feasible release trajectory, then
optimize a 1.3 s full-command suffix. Source case IDs select offline diagnostic
work only; no deployed command is selected by case ID. No native teleport,
learning or expert labels. Original physical constraints are unchanged.
"""
import argparse,hashlib,json,multiprocessing as mp,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_trajectory_recovery_v9 import PureControlTrajectoryRecoveryV9
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain

WORK={}


def initialize(configuration,nominal):
    WORK['controller']=PureControlTrajectoryRecoveryV9(PositionPDInterface(**configuration['interface']),
        nominal,configuration['model_reference_library'])


def rollout(index,nodes,substeps,deadline):
    c=WORK['controller'];source=c.library.records[index];state=c.library.state(source,29)
    limits=c.interface.limits;targets=limits[:,0]+.5*(np.clip(nodes,-1,1)+1)*np.diff(limits).ravel()
    curve=PchipInterpolator([0.,.12,.30,.50,.80,1.10,1.30],np.vstack([state.previous_command_target,targets]),axis=0)
    rows=[];cost=0.;failure=None;stable=0.;longest=0.;previous=state.previous_command_target.copy()
    for k in range(130):
        if time.monotonic()>=deadline:failure=dict(step=k,reason='compute_budget');break
        target=np.clip(curve(k*.01),limits[:,0],limits[:,1])
        try:
            pred=predict_self_contact_spheres(c.model,c.interface,state,target,c.geometry.spheres,
                geometry=c.path_geometry,substeps=substeps)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(step=k,reason='model_rejected',error=repr(error));break
        bad={n:v for n,v in pred['diagnostics'].items() if
            (n=='nonfoot_ground_violation_m' and v>1e-6) or
            (n in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:failure=dict(step=k,reason='physical_path_violation',violations=bad,collision=pred['first_ground_violation']);break
        state=pred['state'];state.previous_command_target=target.copy()
        r=state.root_rotation_world;velocity=r@state.v[:3]
        angles=np.array([np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))])
        omega=float(np.linalg.norm(state.v[3:6]));joint_speed=float(np.max(abs(state.joint_velocity)))
        stable_now=(state.contacts.any() and max(abs(angles))<=np.deg2rad(15) and
            np.linalg.norm(velocity[:2])<=.3 and abs(velocity[2])<=.1 and omega<=1. and joint_speed<=1.)
        stable=stable+.01 if stable_now else 0.;longest=max(stable,longest)
        stage=20.*np.sum(angles**2)+10.*np.sum(velocity**2)+5.*omega**2+.05*joint_speed**2
        cost+=.01*stage+.001*np.sum(((target-previous)/np.diff(limits).ravel())**2)
        previous=target.copy()
        rows.append(dict(step=30+k,q=state.q.copy(),v=state.v.copy(),target=target.copy(),contacts=state.contacts.copy(),
            roll_pitch=angles,root_velocity_world=velocity,omega=omega,joint_speed=joint_speed,stable_s=stable))
    full=failure is None and len(rows)==130
    if rows:
        last=rows[-1];cost+=30.*np.sum(last['roll_pitch']**2)+30.*np.sum(last['root_velocity_world']**2)+10.*last['omega']**2+.2*last['joint_speed']**2
    return dict(nodes=nodes,steps=len(rows),required_steps=130,model_path_complete=full,failure=failure,
        terminal_stable_s=stable,longest_stable_s=longest,cost=float(cost),
        rank=[0 if full else 1,130-len(rows),0 if stable>=.8 else 1,-stable,float(cost)],
        terminal=rows[-1] if rows else None,trajectory=rows if substeps==160 else None,
        substeps=substeps,native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['configuration','fixture-run','standing-run','baseline-metrics','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1200.)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists() or not 120<=a.max_seconds<=1800:raise ValueError('Fresh bounded search required')
    cfg=read(a.configuration);nominal=np.asarray(read(a.fixture_run/'native_command_inputs.json')['default_joint_position_rad'])
    initialize(cfg,nominal);c=WORK['controller'];library=c.library.records
    outcomes={r['training_index']:r for r in read(a.baseline_metrics)['complete5_case_outcomes']}
    indices=[i for i,r in enumerate(library) if outcomes[r['source_case']]['ball5_except_recovery'] and not outcomes[r['source_case']]['recovery']]
    # Previously observed sustained postures are warm starts, not clean expert
    # labels: those native episodes had earlier nonfoot contact.
    with np.load(a.standing_run/'pure_control_commands.npz') as z:
        late=z['tick']>=780;warm=[z['final'][np.flatnonzero(late&(z['lane']==lane))[-1]].copy() for lane in [70,79] if np.any(late&(z['lane']==lane))]
    ends=[nominal,*warm];a.output.mkdir(parents=True)
    def write(name,data):
        path=a.output/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');temp.replace(path)
    pins=[Path(__file__),a.configuration,a.baseline_metrics,a.standing_run/'pure_control_commands.npz',Path(cfg['model_reference_library'])]
    write('inputs.json',dict(source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in pins},
        selected_cases=[library[i]['source_case'] for i in indices],selection='ball5_except_recovery and not recovery in unchanged baseline, with an existing 600 ms model reference',
        fixed_prefix_s=.3,optimized_tail_s=1.3,command_knots=6,command_dimensions=18,
        warm_postures_are_not_expert_labels=True,max_seconds=a.max_seconds,training=False))
    start=time.monotonic();deadline=start+a.max_seconds;results=[]
    try:
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cfg,nominal)) as pool:
            for index in indices:
                source=library[index];limits=c.interface.limits
                write('status.json',dict(state='RUNNING_FULL_COMMAND_TAIL_SEARCH',completed=len(results),total=len(indices),case=source['source_case']))
                seeds=[]
                for end in ends:
                    targets=[]
                    for t in [.12,.30,.50,.80,1.10,1.30]:
                        targets.append(source['trajectory'][min(int(round((.3+t)/.01))-1,59)]['target'] if t<=.30 else end)
                    seeds.append(np.clip(2*(np.asarray(targets)-limits[:,0])/np.diff(limits).ravel()-1,-1,1))
                coarse=[f.result() for f in [pool.submit(rollout,index,n,16,deadline-120) for n in seeds]]
                rng=np.random.default_rng(884025+source['source_case'])
                for generation in range(3):
                    if time.monotonic()>deadline-120:break
                    coarse.sort(key=lambda r:r['rank']);mean=np.asarray([r['nodes'] for r in coarse[:min(5,len(coarse))]]).mean(0)
                    noise=rng.normal(0,[.12,.08,.04][generation],(24,6,18))
                    noise[:,1:]=.5*noise[:,1:]+.5*noise[:,:-1]
                    futures=[pool.submit(rollout,index,np.clip(mean+n,-1,1),16,deadline-120) for n in noise]
                    coarse += [f.result() for f in futures]
                coarse.sort(key=lambda r:r['rank']);best=coarse[0]
                fine=rollout(index,np.asarray(best['nodes']),160,deadline) if time.monotonic()<deadline else None
                result=dict(source_case=source['source_case'],source_key=source['source_key'],
                    coarse_candidates=[{k:v for k,v in r.items() if k not in ['trajectory','terminal']} for r in coarse],
                    fine=fine,full_episode_certified=False,native_execution=False,valid_expert_labels=False)
                results.append(result);write('case_%03d.json'%source['source_case'],result)
        summary=dict(state='COMPLETED_FULL_COMMAND_TAIL_SEARCH',cases=len(results),wall_seconds=time.monotonic()-start,
            fine_complete_cases=[r['source_case'] for r in results if r['fine'] and r['fine']['model_path_complete']],
            fine_stable_tail_cases=[r['source_case'] for r in results if r['fine'] and r['fine']['model_path_complete'] and r['fine']['terminal_stable_s']>=.8],
            native_success=False,full_episode_certified=False,valid_expert_labels=False)
        write('result.json',summary);write('status.json',summary);print(json.dumps(summary),flush=True)
    except BaseException as error:write('status.json',dict(state='FAILED',error=repr(error)));raise


if __name__=='__main__':main()
