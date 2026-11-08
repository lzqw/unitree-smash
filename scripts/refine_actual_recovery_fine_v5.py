"""Optimize complete recovery with native-rate model evaluation throughout.

The preceding coarse search can overestimate survival near impact. This
refinement never uses coarse rollout ranking: every candidate uses 160
substeps. It also gives the release/impact region more command knots, and
concentrates some proposals before the currently observed model failure.
Physical limits and the original end-of-episode stability test are unchanged.
No native restore, RL, controller deployment or expert labels are involved.
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
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.search_current_native_recovery_v4 as base
from scripts.audit_joint_sequence_proposal_v1 import plain


def fine_knots(record):
    return np.asarray(record['search_knots_s'])


def initialize(cfg, nominal, records):
    base.initialize(cfg, nominal, records)
    base.knots_for = fine_knots


def evaluate(key, nodes, deadline):
    return base.rollout(key, nodes, 160, deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'configuration', 'run', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=3000.)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        return h.hexdigest()
    parent = read(args.parent/'inputs.json')
    outcome = read(args.parent/'result.json')
    if (args.output.exists() or not 900 <= args.max_seconds <= 3600
        or outcome['state'] != 'COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH'
        or read(args.run/'job.json')['status'] != 'SUCCEEDED'):
        raise ValueError('Fresh bounded refinement and complete source evidence required')
    for path, digest in parent['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed source of original search: ' + path)
    if parent['source_sha256'].get(str(args.configuration.resolve())) != sha(args.configuration):
        raise ValueError('Refinement must retain the original interface')
    cfg = read(args.configuration)
    nominal = read(args.run/'native_command_inputs.json')['default_joint_position_rad']
    records = parent['states']
    with np.load(args.run/'pure_control_commands.npz') as z:
        for key, record in records.items():
            ids = np.flatnonzero((z['lane']==record['case']) & (z['tick']==record['tick']))
            if len(ids)!=1 or any(not np.array_equal(z[k][ids[0]],record[k]) for k in ['q','v','previous','ground_force']):
                raise ValueError('Planning state differs from recorded native observation: '+key)
    source_plans = {key:read(args.parent/('state_'+key.replace(':','_')+'.json')) for key in records}
    originals = {key:np.asarray(parent['knots_s'][key]) for key in records}
    for key, record in records.items():
        # Timing freedom in the first 0.8 seconds, plus full-episode controls.
        record['search_knots_s'] = sorted(set(originals[key].tolist()+[.01,.03,.06,.16,.35,.65]))
    initialize(cfg, nominal, records)
    limits = base.WORK['c'].interface.limits
    span = np.diff(limits).ravel()
    rng = {key:np.random.default_rng(889000+i) for i,key in enumerate(records)}
    pins = dict(parent['source_sha256'])
    paths = [Path(__file__),args.parent/'inputs.json',args.parent/'result.json']
    paths += [args.parent/('state_'+key.replace(':','_')+'.json') for key in records]
    paths += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins.update({str(p.resolve()):sha(p) for p in paths})
    args.output.mkdir(parents=True)
    def write(name, value):
        path = args.output/name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    write('inputs.json',dict(source_sha256=pins,states=records,
        knots_s={key:fine_knots(r) for key,r in records.items()},max_seconds=args.max_seconds,workers=8,
        all_candidates_substeps=160,first_command_free=True,dimensions=18,
        future_native_actions_used=False,training=False,native_state_teleport=False,
        hardware_changed=False,acceptance_changed=False,seed_resampling_is_not_behavior_identity=True))
    started = time.monotonic()
    deadline = started+args.max_seconds
    history = {key:[] for key in records}
    best = {}
    completed = 0
    def accept(key, row):
        nonlocal completed
        completed += 1
        history[key].append({k:v for k,v in row.items() if k not in ['trajectory','terminal']})
        if key not in best or row['rank'] < best[key]['rank']:
            best[key] = row
            write('state_'+key.replace(':','_')+'.json',dict(state_key=key,
                initial_actual_state=records[key],fine=row,
                all_candidates_evaluated_at_substeps=160,full_episode_certified=False,
                native_success=False,valid_expert_labels=False))
        write('status.json',dict(state='REFINING_ACTUAL_RECOVERY_FINE',pid=os.getpid(),
            completed_candidates=completed,elapsed_s=time.monotonic()-started,
            best={k:dict(steps=v['steps'],model_path_complete=v['model_path_complete'],
                terminal_stable_s=v['terminal_stable_s'],failure=v['failure']) for k,v in best.items()}))
    write('status.json',dict(state='REFINING_ACTUAL_RECOVERY_FINE',pid=os.getpid(),completed_candidates=0))
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),
        initializer=initialize,initargs=(cfg,nominal,records)) as pool:
        jobs={}
        for key, record in records.items():
            fine=source_plans[key]['fine']
            if fine['substeps']!=160 or fine['failure'] is None:
                raise ValueError('This refinement targets incomplete fine trajectories')
            nodes=np.clip(PchipInterpolator(originals[key],fine['nodes'],axis=0)(fine_knots(record)),-1,1)
            jobs[pool.submit(evaluate,key,nodes,deadline)]=key
        for future in as_completed(jobs):
            accept(jobs[future],future.result())
        for generation in range(8):
            if time.monotonic()>deadline-120:
                break
            jobs={}
            for key, record in records.items():
                elite=sorted(history[key],key=lambda row:row['rank'])[:3]
                knots=fine_knots(record)
                for proposal in range(8):
                    seed=elite[proposal%len(elite)]
                    # Propose coherent full-body motions; no small-residual
                    # execution restriction is imposed. All18 remain free.
                    sigma=[.12,.3,.6,.9][proposal%4]
                    physical=rng[key].normal(0,.35*sigma,(len(knots),18))
                    extension=rng[key].normal(0,sigma,(len(knots),1))+rng[key].normal(0,.4*sigma,(len(knots),4))
                    physical[:,8:12]+=extension
                    physical[:,4:8]-=.5*extension
                    physical[:,12:]+=rng[key].normal(0,sigma,(len(knots),6))
                    physical[1:]=.65*physical[1:]+.35*physical[:-1]
                    if proposal<4:
                        failure_s=seed['steps']*.01
                        window=np.maximum(0,1-abs(knots-max(0,failure_s-.2))/.7)
                        physical*=window[:,None]
                    nodes=np.clip(np.asarray(seed['nodes'])+2*physical/span,-1,1)
                    jobs[pool.submit(evaluate,key,nodes,deadline)]=key
            for future in as_completed(jobs):
                accept(jobs[future],future.result())
    for path,digest in pins.items():
        if sha(path)!=digest:
            raise ValueError('Changed refinement input '+path)
    results=[]
    for key,row in best.items():
        results.append(dict(state_key=key,steps=row['steps'],model_path_complete=row['model_path_complete'],
            terminal_stable_s=row['terminal_stable_s'],window_violation=row['window_violation'],failure=row['failure']))
        path=args.output/('state_'+key.replace(':','_')+'.json')
        value=read(path)
        value['fine_candidate_history']=history[key]
        write(path.name,value)
    summary=dict(state='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH',results=results,
        completed_candidates=completed,all_candidates_substeps=160,
        wall_seconds=time.monotonic()-started,native_success=False,valid_expert_labels=False)
    write('result.json',summary)
    write('status.json',summary)
    print(json.dumps(plain(summary)),flush=True)


if __name__=='__main__':
    main()
