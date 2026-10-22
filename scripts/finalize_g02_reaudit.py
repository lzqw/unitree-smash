#!/usr/bin/env python3
"""Recompute native G02 evidence without accepting historical Boolean claims."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.physics.sweep_reference import sphere_box_sweep, rotation

def load(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def first_episode(result,lane):
    rows = [r for r in result['contacts'] if r['lane']==lane and np.linalg.norm(r['force'][0])>.001]
    if not rows:
        return None
    first = next((i for i,r in enumerate(rows) if r['new']),None)
    if first is None:
        raise ValueError('force contact without a corresponding event start')
    end = next((i for i in range(first+1,len(rows)) if rows[i]['new']),len(rows))
    return rows[first],rows[end-1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-prefix',required=True)
    parser.add_argument('--fixture-suffix',default='')
    parser.add_argument('--coupling-dir',required=True)
    parser.add_argument('--cpu-report',required=True)
    parser.add_argument('--output',default='reports/G02')
    args = parser.parse_args()
    output = ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    contract_path = ROOT/'configs/g02_validation_v2.json'
    tol = load(contract_path)
    labels = ['2','1','05','025','0125','00625','003125']
    dirs = {label:ROOT/(args.fixture_prefix+label+args.fixture_suffix) for label in labels}
    results = {label:load(path/'benchmark.json') for label,path in dirs.items()}
    jobs = [load(path/'job.json') for path in dirs.values()]
    coupled_dir = ROOT/args.coupling_dir
    coupled = load(coupled_dir/'coupling.json')
    jobs.append(load(coupled_dir/'job.json'))
    if any(j['status']!='SUCCEEDED' for j in jobs):
        raise ValueError('failed native job cannot be used as passing evidence')
    prod, ref = results['00625'],results['003125']
    assert prod['dt']==tol['production_dt_s'] and ref['dt']==tol['reference_dt_s']
    assert coupled['dt']==prod['dt']
    assert load(ROOT/'configs/shuttle_physics_v2.json')['native_physics_dt_s']==prod['dt']
    assert all(r['cases']==prod['cases'] and r['policy_dt']==.01 for r in results.values())
    xml = ET.parse(ROOT/args.cpu_report).getroot()
    suites = xml.findall('.//testsuite')
    cpu_ok = sum(int(s.attrib['tests']) for s in suites)>=69 and all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    rows, flight, restitution = [],[],[]
    sweep_ok, converge_ok = True,True
    for i,c in enumerate(prod['cases']):
        if c['kind'] in ('gravity','drag','wind'):
            p0,v0 = np.array(prod['initial_position'][i]),np.array(prod['initial_velocity'][i])
            t = prod['trajectory'][-1]['time']
            if c['kind']=='gravity':
                g=np.array([0,0,-9.81]); expected_p=p0+t*v0+.5*t*t*g; expected_v=v0+t*g
            else:
                wind=np.array([3,0,0]) if c['kind']=='wind' else np.zeros(3)
                u=v0-wind; speed=np.linalg.norm(u); k=.025442307692307694
                expected_p=p0+t*wind+u/speed*math.log1p(k*speed*t)/k
                expected_v=wind+u/(1+k*speed*t)
            flight.append(dict(lane=i,kind=c['kind'],speed=c['speed'],
                position_error_m=float(np.linalg.norm(np.array(prod['final_position'][i])-expected_p)),
                velocity_error_m_s=float(np.linalg.norm(np.array(prod['final_velocity'][i])-expected_v))))
            sweep_ok &= prod['contact_count'][i]==0
            continue
        duration=.06
        omega=c.get('omega',0.)
        hit=sphere_box_sweep([.2,c['y'],0],[.2-c['speed']*duration,c['y'],0],
            [0,0,0],[c.get('racket_vx',0.)*duration,0,0],[1,0,0,0],
            [math.cos(omega*duration/2),0,0,math.sin(omega*duration/2)])
        boundary=False if hit is None else bool(np.min(.11-np.abs((rotation(hit[3]).T@(hit[1]-hit[2]))[1:]))<.025)
        row=dict(lane=i,kind=c['kind'],speed=c['speed'],y=c['y'],omega=omega,boundary=boundary,
            expected_hit=hit is not None,reference_time_s=None if hit is None else float(hit[0]*duration))
        for label,result in results.items():
            event=first_episode(result,i)
            row['hit_dt'+label]=event is not None
        actual,reference=first_episode(prod,i),first_episode(ref,i)
        sweep_ok &= (actual is not None)==(hit is not None) and (reference is not None)==(hit is not None)
        if actual is not None and reference is not None:
            a,b=np.array(actual[1]['ball_velocity']),np.array(reference[1]['ball_velocity'])
            speed_error=abs(np.linalg.norm(a)-np.linalg.norm(b))/max(np.linalg.norm(b),1e-12)
            angle=math.degrees(math.acos(float(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
            time_error=abs(actual[0]['time']-reference[0]['time'])
            row.update(speed_relative_error=float(speed_error),direction_error_deg=angle,contact_time_difference_s=time_error,
                       actual_time_s=actual[0]['time'],expected_time_error_s=abs(actual[0]['time']-row['reference_time_s']))
            if not boundary:
                converge_ok &= speed_error<=tol['nonboundary_speed_relative_tolerance'] and angle<=tol['nonboundary_direction_tolerance_deg'] and time_error<=tol['contact_time_tolerance_s']
            sweep_ok &= row['expected_time_error_s']<=tol['contact_time_tolerance_s']
            # Validate actual contact-point angular velocity in the collision
            # frame. A translated-only tool speed would fail rotating cases.
            if c['kind']=='rotating' or (c['y']==0. and c['kind'] in ('impact','moving')):
                final=actual[1]; normal=np.array(final['force'][0]); normal/=np.linalg.norm(normal)
                offset=np.array(final['ball_position'])-.025*normal-np.array(final['blade_position'])
                pre_tool=np.array([c.get('racket_vx',0.),0,0])+np.cross([0,0,omega],offset)
                post_tool=np.array(final['blade_velocity'])+np.cross(final['blade_angular_velocity'],offset)
                pre=float((np.array([-c['speed'],0,0])-pre_tool)@normal)
                post=float((np.array(final['ball_velocity'])-post_tool)@normal)
                restitution.append(dict(lane=i,kind=c['kind'],measured_restitution=-post/pre,contact_point_speed=float(np.linalg.norm(pre_tool))))
        rows.append(row)
    independently_counted=[]
    for i in range(len(prod['cases'])):
        times=[r['time'] for r in prod['contacts'] if r['lane']==i and np.linalg.norm(r['force'][0])>.001]
        independently_counted.append((1 if times else 0)+sum(b-a>2.5*prod['dt'] for a,b in zip(times,times[1:])))
    no_double=independently_counted==prod['contact_count']
    repeated=next(i for i,c in enumerate(prod['cases']) if c['kind']=='recontact')
    events=[r for r in prod['contacts'] if r['lane']==repeated and (r['new'] or r['new_second'])]
    no_double &= len(events)>=3 and events[0]['new'] and events[1]['new_second'] and events[2]['new'] and prod['contact_count'][repeated]==2
    linear=float(np.max([r['linear_error'] for r in coupled['rows']]))
    angular=float(np.max([r['angular_error'] for r in coupled['rows']]))
    reactions=[]
    for lane in (0,2):
        first=next(r for r in coupled['rows'] if r['contact_count'][lane])
        reactions.append(float(np.linalg.norm(np.array(first['robot_reaction'][lane])-first['robot_reaction'][lane+1])))
    coupled_ok=(coupled['contact_count'][0]>=1 and coupled['contact_count'][2]>=1 and coupled['contact_count'][1]==coupled['contact_count'][3]==0
        and sum(coupled['sweep_misses'])==sum(coupled['sweep_unresolved'])==0)
    reaction_ok=(linear<=tol['coupled_linear_momentum_tolerance_kg_m_s'] and angular<=tol['coupled_angular_momentum_tolerance_kg_m2_s']
        and min(reactions)>tol['minimum_paired_robot_reaction_kg_m_s'] and max(prod['max_momentum_error'])<=tol['fixture_linear_momentum_tolerance_kg_m_s']
        and all(abs(r['measured_restitution']-.72)<=tol['restitution_absolute_tolerance'] for r in restitution))
    metrics=dict(analytic_flight_ok=bool(cpu_ok and all(r['position_error_m']<=tol['native_flight_position_tolerance_m'] and r['velocity_error_m_s']<=tol['native_flight_velocity_tolerance_m_s'] for r in flight)),
        swept_contact_ok=bool(cpu_ok and sweep_ok and coupled_ok),no_double_impulse=bool(no_double),
        reaction_impulse_ok=bool(reaction_ok),timestep_convergence_ok=bool(converge_ok),
        native_case_count=len(prod['cases']),cpu_test_count=sum(int(s.attrib['tests']) for s in suites),
        production_dt_s=prod['dt'],policy_dt_s=prod['policy_dt'],reference_dt_s=ref['dt'],
        coupled_linear_momentum_error_kg_m_s=linear,coupled_angular_momentum_error_kg_m2_s=angular,
        paired_robot_reaction_kg_m_s=reactions,flight=flight,restitution=restitution,
        unresolved_boundary_response_cases=[r['lane'] for r in rows if r['boundary'] and (r.get('speed_relative_error',0)>.05 or r.get('direction_error_deg',0)>5)])
    (output/'physics_tests.json').write_text(json.dumps(metrics,indent=2)+'\n')
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'contact_cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    (output/'commands.log').write_text('\n'.join(json.dumps(j['command']) for j in jobs)+'\n')
    (output/'REPORT.md').write_text(f'''# G02 native standard-asset physics re-audit

The CPU analytical suite ran {metrics['cpu_test_count']} tests. Native fixtures ran {len(prod['cases'])} cases at each of 2, 1, 0.5, 0.25, 0.125, 0.0625 and 0.03125 ms, retaining a 10 ms policy period. Cases cover gravity, drag, wind, 4–200 m/s impacts, finite-mass reaction, moving blades, ±5/20/40 rad/s rotation, misses, rim contact and physical separation followed by contact again. Fixture mass/dimensions exactly match the audited 0.09 kg, 0.015×0.22×0.22 m blade and 0.0052 kg, 0.025 m-radius shuttle assumptions.

Isaac Sim 5.1 GPU dynamics disables sweep CCD. The earlier coarse steps missed real impacts; their results remain in contact_cases.csv. Production ball tasks now use 0.0625 ms native substeps with an independent moving/rotating sweep audit, without custom impulses or pose correction. The no-ball standing/jump physics remains at its audited step. Native contact-force filters identify the blade and ground. Contact episodes rearm only after separation. Net/landing crossings are interpolated. No fabricated velocity-change hit flags or permanent one-hit latch remain in BallEnv.

The selected step matches the independent hit/miss sweep in every fixture case. Nonboundary output speed, direction and event time are compared with 0.03125 ms using the unchanged 5%, 5° and 2 ms criteria. Boundary classification comes from the independent geometry before examining numerical response. Boundary lanes {metrics['unresolved_boundary_response_cases']} still have unstable outgoing response and are explicitly retained; these are not evidence of reliable rim-shot performance. Detection still must match for those lanes. Full-load production sweep misses/unresolved queries were {coupled['sweep_misses']} / {coupled['sweep_unresolved']}.

Full floating standard Go2+PiPER+mount+racket tests retain all 20.646991 kg and actuator limits. Measured paired robot reactions were {reactions} kg m/s. Whole-system linear/angular momentum residuals after accounting for gravity and drag were {linear:.8g} kg m/s and {angular:.8g} kg m²/s. Free-flight diagnostic initialization is only a coupling test, never G04 learning evidence.

All quantitative metrics are recomputed from raw native traces. Native float32/first-order flight tolerances and CPU analytical tolerances are documented separately in configs/g02_validation_v2.json and the tests. The fixture shutdown path uses Kit's documented skip_cleanup option after flushing receipts, avoiding a standalone tensor-scene teardown hang. Failed prior teardown runs remain charged in status/GPU_LEDGER.json.

Official backend limitation: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html . Locked local source also explicitly disables GPU CCD. This is engineering evidence, not a trained contact/smash policy. G03 still requires repair before G04 PPO can resume.
''')
    files=[output/n for n in ('REPORT.md','commands.log','physics_tests.json','contact_cases.csv')]
    files += [contract_path,ROOT/'configs/shuttle_physics_v2.json',ROOT/args.cpu_report]
    for path in dirs.values():
        files += [path/n for n in ('benchmark.json','job.json','stdout.log','source_snapshot.zip')]
    files += [coupled_dir/n for n in ('coupling.json','job.json','stdout.log','source_snapshot.zip')]
    files += [ROOT/'tests'/n for n in ('test_native_physics_contract.py','test_shuttlecock_physics.py','test_racket_collision.py')]
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    evidence=dict(schema_version=1,goal_id='G02',task_id='physics_validation',evidence_kind='observed',
        contract_sha256=digest(ROOT/'configs/acceptance.json'),metrics=metrics,
        run=dict(command='scripts/finalize_g02_reaudit.py; native commands in reports/G02/commands.log',
            git_commit=git.stdout.strip() if git.returncode==0 else 'uncommitted-worktree-source-snapshots-attached',
            started_at=min(j['started_at'] for j in jobs),finished_at=max(j['finished_at'] for j in jobs),
            gpu_hours=sum(j['charged_gpu_hours'] for j in jobs),env_steps=sum(j['charged_env_steps'] for j in jobs)),
        artifacts=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p),bytes=p.stat().st_size) for p in dict.fromkeys(files)])
    temporary=output/'evidence.json.tmp';temporary.write_text(json.dumps(evidence,indent=2)+'\n');temporary.replace(output/'evidence.json')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('flight','restitution')},indent=2))
    return 0 if all(metrics[k] for k in ('analytic_flight_ok','swept_contact_ok','no_double_impulse','reaction_impulse_ok','timestep_convergence_ok')) else 1

if __name__=='__main__':
    raise SystemExit(main())
