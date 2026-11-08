"""Search recovery sequences through sustained support, from actual release states.

The whole 3.2-second command sequence remains adjustable. Prior 1.6-second
plans are warm starts; terminal stability is prioritized over a standing-pose
cost. This is CPU model search, not native success or expert demonstration.
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
from scripts.pure_control_long_trajectory_recovery_v11 import PureControlLongTrajectoryRecoveryV11
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain

KNOTS = [.08,.18,.30,.45,.60,.90,1.20,1.60,2.0,2.6,3.2]
STEPS = 320
WORK = {}


def initialize(configuration, nominal):
    WORK['controller'] = PureControlLongTrajectoryRecoveryV11(
        PositionPDInterface(**configuration['interface']), nominal,
        configuration['model_reference_library'])


def rollout(index,nodes,substeps,deadline):
    c=WORK['controller'];source=c.library.records[index];state=c.library.state(source,-1)
    limits=c.interface.limits;targets=limits[:,0]+.5*(np.clip(nodes,-1,1)+1)*np.diff(limits).ravel()
    curve=PchipInterpolator([0., *KNOTS],np.vstack([state.previous_command_target,targets]),axis=0)
    rows=[];cost=0.;failure=None;stable=0.;longest=0.;previous=state.previous_command_target.copy()
    for k in range(STEPS):
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
        rows.append(dict(step=k,q=state.q.copy(),v=state.v.copy(),target=target.copy(),contacts=state.contacts.copy(),
            roll_pitch=angles,root_velocity_world=velocity,omega=omega,joint_speed=joint_speed,stable_s=stable))
    full=failure is None and len(rows)==STEPS
    if rows:
        last=rows[-1];cost+=30.*np.sum(last['roll_pitch']**2)+30.*np.sum(last['root_velocity_world']**2)+10.*last['omega']**2+.2*last['joint_speed']**2
    return dict(nodes=nodes,steps=len(rows),required_steps=STEPS,model_path_complete=full,failure=failure,
        terminal_stable_s=stable,longest_stable_s=longest,cost=float(cost),
        rank=[0 if full else 1,STEPS-len(rows),0 if stable>=.8 else 1,-stable,float(cost)],
        terminal=rows[-1] if rows else None,trajectory=rows if substeps==160 else None,
        substeps=substeps,native_execution=False,valid_expert_labels=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'fixture-run', 'prior-search', 'full-model-results', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=1800.)
    args = parser.parse_args()
    if args.output.exists() or not 900 <= args.max_seconds <= 3600:
        raise ValueError('Fresh bounded search required')
    read = lambda p: json.loads(Path(p).read_text())
    cfg = read(args.configuration)
    nominal = np.asarray(read(args.fixture_run / 'native_command_inputs.json')['default_joint_position_rad'])
    initialize(cfg, nominal)
    c = WORK['controller']
    # Selection is retrospective experiment scheduling. Runtime matching uses
    # the causal state, not these source IDs or final outcomes.
    selected = sorted(r['source_case'] for f in args.full_model_results.glob('case_*.json')
        for r in [read(f)] if r['failure'] is not None and r['steps'] >= 160)
    indices = [i for i, r in enumerate(c.library.records) if r['source_case'] in selected]
    if not indices or len(indices) != len(selected):
        raise ValueError('Expected all late-failing source states in the model library')
    limits = c.interface.limits
    spans = np.diff(limits).ravel()
    normalize = lambda target: np.clip(2 * (np.asarray(target)-limits[:, 0])/spans-1, -1, 1)
    args.output.mkdir(parents=True)

    def write(name, value):
        path = args.output / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
        temp.replace(path)

    sources = [Path(__file__), args.configuration, Path(cfg['model_reference_library']),
               args.fixture_run/'native_command_inputs.json']
    sources += sorted(args.full_model_results.glob('case_*.json'))
    sources += [args.prior_search/('case_%03d.json' % k) for k in selected]
    sources += [Path(m.__file__) for n, m in list(sys.modules.items())
                if n.startswith('scripts.') and getattr(m, '__file__', None)]
    pins = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write('inputs.json', dict(source_sha256=pins, selected_cases=selected, max_seconds=args.max_seconds,
        optimized_horizon_s=3.2, command_knots=KNOTS, dimensions=18, fixed_prefix_s=0., workers=8,
        terminal_stable_requirement_s=.8, original_physical_limits=True,
        native_success=False, training=False, valid_expert_labels=False))
    started = time.monotonic()
    deadline = started + args.max_seconds
    candidates = {i: [] for i in indices}
    rngs = {i: np.random.default_rng(885100+c.library.records[i]['source_case']) for i in indices}
    completed = 0
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=(cfg, nominal)) as pool:
        pending = []
        for i in indices:
            r = c.library.records[i]
            prior = read(args.prior_search/('case_%03d.json' % r['source_case']))['fine']
            if not prior['model_path_complete']:
                raise ValueError('Prior complete prefix required')
            prefix = np.asarray(prior['nodes'])
            final_target = np.asarray(prior['trajectory'][-1]['target'])
            final_posture = np.asarray(prior['trajectory'][-1]['q'])[c.model.q_indices]
            for terminal in [final_target, final_posture, nominal, .5*(final_posture+nominal)]:
                suffix = np.stack([normalize(terminal)]*3)
                nodes = np.vstack([prefix, suffix])
                pending.append((i, pool.submit(rollout, i, nodes, 16, deadline-600.)))
        for i, future in pending:
            candidates[i].append(future.result())
            completed += 1
        for generation, sigma in enumerate([.24, .16, .10, .06]):
            if time.monotonic() >= deadline-600:
                break
            pending = []
            for i in indices:
                ranked = sorted(candidates[i], key=lambda r:r['rank'])
                # Sample around one actual elite, not the mean of potentially
                # incompatible contact sequences.
                elite = ranked[:min(4, len(ranked))]
                rng = rngs[i]
                for j in range(24):
                    center = np.asarray(elite[j % len(elite)]['nodes'])
                    physical = rng.normal(0, .25*sigma, (len(KNOTS), 18))
                    extension = rng.normal(0, sigma, (len(KNOTS), 1)) + rng.normal(0, .4*sigma, (len(KNOTS), 4))
                    physical[:,8:12] += extension
                    physical[:,4:8] -= .5*extension
                    physical[:,:4] += rng.normal(0, .5*sigma, (len(KNOTS),1))
                    physical[:,12:] += rng.normal(0, sigma, (len(KNOTS),6))
                    physical[1:] = .5*physical[1:] + .5*physical[:-1]
                    # Warm-start bias, not an action bound: early commands
                    # remain adjustable while more search explores loading.
                    physical[:5] *= .2
                    nodes = np.clip(center + 2*physical/spans, -1, 1)
                    pending.append((i, pool.submit(rollout, i, nodes, 16, deadline-600.)))
            for i, future in pending:
                candidates[i].append(future.result())
                completed += 1
            write('status.json', dict(state='SEARCHING_SUSTAINED_SUPPORT', pid=os.getpid(),
                generation=generation+1, completed_candidates=completed, elapsed_s=time.monotonic()-started))
        fine_jobs = []
        for i in indices:
            ranked = sorted(candidates[i], key=lambda r:r['rank'])
            # Check more than a single coarse optimum against the fine plant.
            for rank, candidate in enumerate(ranked[:3]):
                fine_jobs.append((i, rank, pool.submit(rollout, i, np.asarray(candidate['nodes']), 160, deadline)))
        fines = {i: [] for i in indices}
        for i, rank, future in fine_jobs:
            fines[i].append(dict(coarse_rank=rank, result=future.result()))
        output = []
        for i in indices:
            source = c.library.records[i]
            fine = min(fines[i], key=lambda r:r['result']['rank'])['result']
            result = dict(source_case=source['source_case'], query=source['query'],
                source_key=source['source_key'], fine=fine,
                coarse_candidates=[{k:v for k,v in r.items() if k not in ['trajectory','terminal']}
                    for r in sorted(candidates[i], key=lambda r:r['rank'])],
                fine_alternatives=[{k:v for k,v in r['result'].items() if k not in ['trajectory','terminal']}
                    for r in fines[i]], native_success=False, full_episode_certified=False, valid_expert_labels=False)
            write('case_%03d.json' % source['source_case'], result)
            output.append(result)
    for name, expected in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed search input ' + name)
    summary = dict(state='COMPLETED_SUSTAINED_SUPPORT_SEARCH', completed_candidates=completed,
        wall_seconds=time.monotonic()-started,
        fine_path_complete_cases=[r['source_case'] for r in output if r['fine']['model_path_complete']],
        fine_stable_08s_cases=[r['source_case'] for r in output if r['fine']['model_path_complete'] and r['fine']['terminal_stable_s']>=.8],
        native_success=False, full_episode_certified=False, valid_expert_labels=False)
    write('result.json', summary)
    write('status.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
