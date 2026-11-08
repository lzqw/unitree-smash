"""Release the fixed post-hit prefix and optimize the full recovery from its actual start.

A completed partial model trajectory is only a warm start. Its early commands
are now adjustable too. Source prefix/suffix commands are reconstructed before
resampling to full-recovery knots. Native state, hardware and acceptance stay
unchanged. This is bounded offline control search, not a native success claim.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.search_current_native_recovery_v4 import initialize,rollout,knots_for,WORK
from scripts.audit_joint_sequence_proposal_v1 import plain


def interleaved(proposals):
    """Yield one proposal per nonempty state in each round, preserving its order."""
    for index in range(max(map(len,proposals.values()),default=0)):
        for key,rows in proposals.items():
            if index<len(rows):
                yield key,rows[index]


def timed_rollout(key,nodes,deadline,submitted_at):
    started=time.monotonic();cpu_started=time.process_time()
    result=rollout(key,nodes,160,deadline)
    result['worker_wall_seconds']=time.monotonic()-started
    result['worker_cpu_seconds']=time.process_time()-cpu_started
    result['queue_wait_seconds']=started-submitted_at
    return result


def source_targets(source, parent, times, limits):
    fine = source['fine']
    count = int(fine.get('prefix_steps', 0)) if fine.get('nodes_describe_suffix_only') else 0
    if count:
        split = source['model_split_state']
        if split.get('state_origin') != 'model_prefix_not_native':
            raise ValueError('Expected an explicitly model-predicted source split')
        if split['tick'] != source['initial_actual_state']['tick'] + count:
            raise ValueError('Incorrect source split clock')
        if not 0 < count < len(fine['trajectory']):
            raise ValueError('Invalid checked source prefix')
        last = fine['trajectory'][count - 1]
        for key, field in [('q', 'q'), ('v', 'v'), ('previous', 'target'), ('ground_force', 'ground_force')]:
            if not np.array_equal(split[key], last[field]):
                raise ValueError('Source split does not match checked prefix')
        knots = knots_for(split)
    else:
        knots = np.asarray(parent['knots_s'][source['state_key']], float)
    if len(knots) != len(fine['nodes']):
        raise ValueError('Source knot and node counts differ')
    times = np.asarray(times, float)
    normalized = PchipInterpolator(knots, np.asarray(fine['nodes']), axis=0)(times - count * .01)
    values = limits[:, 0] + .5 * (np.clip(normalized, -1., 1.) + 1.) * np.diff(limits).ravel()
    if count:
        checked = np.asarray([r['target'] for r in fine['trajectory'][:count]])
        before = times < count * .01 - 1e-10
        for joint in range(18):
            values[before, joint] = np.interp(times[before], np.arange(count) * .01, checked[:, joint])
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['search','configuration','fixture-run','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--states',nargs='+',required=True,help='case:actual_tick; every recovery command is adjustable')
    p.add_argument('--max-seconds',type=int,default=2400)
    p.add_argument('--check-only',action='store_true',help='Validate source snapshots and scheduling without searching or writing output')
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    if not 600<=a.max_seconds<=3600:raise ValueError('Bounded fine search duration required')
    if a.output.exists() or read(a.search/'result.json')['state']!='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
        raise ValueError('Fresh output and completed source planning required')
    for f,h in read(a.search/'inputs.json')['source_sha256'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed source '+f)
    parent=read(a.search/'inputs.json')
    if parent['source_sha256'].get(str(a.configuration.resolve())) != hashlib.sha256(a.configuration.read_bytes()).hexdigest():
        raise ValueError('Keep the original physical and command configuration')
    cfg=read(a.configuration)
    nominal=read(a.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    if read(a.fixture_run/'job.json')['status'] != 'SUCCEEDED':
        raise ValueError('Completed native actual-state evidence required')
    records={};sources={};prefixes={};seed_errors={}
    files=[Path(__file__),a.configuration,a.fixture_run/'job.json',
        a.fixture_run/'native_command_inputs.json',a.fixture_run/'pure_control_commands.npz',
        a.search/'inputs.json',a.search/'result.json',Path(cfg['model_reference_library'])]
    with np.load(a.fixture_run/'pure_control_commands.npz') as data:
        native={k:data[k].copy() for k in ['lane','tick','q','v','previous','ground_force']}
    for key in a.states:
        case,tick=map(int,key.split(':'))
        path=a.search/('state_'+key.replace(':','_')+'.json');source=read(path)
        if key in records or source['native_success'] or source['valid_expert_labels']:
            raise ValueError('Distinct model-only seed required')
        fine=source['fine'];initial=source['initial_actual_state'];trace=fine['trajectory']
        if fine['substeps'] != 160 or len(trace)<2 or not 100<=800-tick<=600:
            raise ValueError('Checked seed and original 1-6 second horizon required')
        if initial['case']!=case or initial['tick']!=tick or source['state_key']!=key:
            raise ValueError('Wrong source state')
        if any(row['step']!=i for i,row in enumerate(trace)):
            raise ValueError('Discontinuous source command sequence')
        ids=np.flatnonzero((native['lane']==case)&(native['tick']==tick))
        if len(ids)!=1 or any(not np.array_equal(native[k][ids[0]],initial[k])
                for k in ['q','v','previous','ground_force']):
            raise ValueError('Start must exactly equal recorded native observation')
        if not initial['phase']['owns_final_recovery_command']:
            raise ValueError('No causal recovery ownership at initial state')
        records[key]=dict(initial,state_origin='recorded_native_initial_state')
        prefixes[key]=0;sources[key]=source;files.append(path)
    initialize(cfg,nominal,records);c=WORK['c'];limits=c.interface.limits;span=np.diff(limits).ravel()
    norm=lambda q:np.clip(2*(np.asarray(q)-limits[:,0])/span-1,-1,1)
    for key,source in sources.items():
        trace=source['fine']['trajectory']
        reconstructed=source_targets(source,parent,np.arange(len(trace))*.01,limits)
        error=float(np.max(abs(reconstructed-np.asarray([r['target'] for r in trace]))))
        if error>1e-9:raise ValueError('Cannot reconstruct source command clock '+key)
        seed_errors[key]=error
    if a.check_only:
        preview=list(interleaved({key:list(range(9)) for key in records}))
        if len(preview)!=9*len(records) or any([key for key,_ in preview[i:i+len(records)]]!=list(records)
                for i in range(0,len(preview),len(records))):
            raise ValueError('Candidate rounds do not cover all states')
        print(json.dumps(dict(state='INPUT_AND_SCHEDULING_CHECKED',states=list(records),
            initial_candidate_count=len(preview),first_rounds=preview[:2*len(records)],
            native_execution=False,model_rollout_executed=False,training=False,first_command_free=True,source_command_reconstruction_max_error_rad=seed_errors)))
        return
    a.output.mkdir(parents=True)
    def write(name,value):
        f=a.output/name;t=f.with_suffix('.tmp');t.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');t.replace(f)
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    write('inputs.json',dict(source_sha256=pins,model_split_states=records,prefix_steps=prefixes,
        original_actual_initial_states={k:s['initial_actual_state'] for k,s in sources.items()},
        native_state_teleport=False,future_native_commands_used=False,training=False,
        workers=4,max_seconds=a.max_seconds,knots_s={k:knots_for(r) for k,r in records.items()},all_candidates_substeps=160,physical_limits_changed=False,
        first_command_free=True,fixed_recovery_prefix_steps=0,source_command_reconstruction_max_error_rad=seed_errors,
        seed_is_not_an_executable_certificate=True,
        scheduling='round_robin_candidates_across_states',
        equal_cpu_time_guaranteed=False,progress_every_completed_candidates=8))
    started=time.monotonic();deadline=started+a.max_seconds;candidates={k:[] for k in records}
    rng={k:np.random.default_rng(887000+i) for i,k in enumerate(records)};completed=0
    generation=0
    def progress():
        write('status.json',dict(state='SEARCHING_FULL_RECOVERY_FROM_ACTUAL_RELEASE',pid=os.getpid(),
            generation=generation,completed_candidates=completed,elapsed_s=time.monotonic()-started,
            per_state={key:dict(completed_candidates=len(rows),
                budget_limited=sum((r.get('failure') or {}).get('reason')=='compute_budget' for r in rows),
                worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),
                worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows),
                maximum_queue_wait_seconds=max((r['queue_wait_seconds'] for r in rows),default=0.),
                complete_model_paths=sum(r['model_path_complete'] for r in rows),
                complete_stable_model_paths=sum(r['model_path_complete'] and r['terminal_stable_s']>=.8 for r in rows),
                best_steps=max((r['steps'] for r in rows),default=0)) for key,rows in candidates.items()}))
    progress()
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cfg,nominal,records)) as pool:
        def evaluate_proposals(proposals):
            nonlocal completed
            submitted={key:len(candidates[key]) for key in records};jobs={}
            for key,nodes in interleaved(proposals):
                jobs[pool.submit(timed_rollout,key,nodes,deadline,time.monotonic())]=(key,submitted[key]);submitted[key]+=1
            for job in as_completed(jobs):
                key,index=jobs[job];row=job.result();row['proposal_index']=index
                candidates[key].append(row);completed+=1
                if completed%8==0:progress()
            progress()
        proposals={key:[] for key in records}
        for key,record in records.items():
            knots=knots_for(record);source=sources[key]
            original_tail=norm(source_targets(source,parent,knots,limits))
            proposals[key].append(original_tail)
            q=np.asarray(record['q'])[c.model.q_indices];previous=np.asarray(record['previous'])
            for goal in [q,previous,np.asarray(nominal),.5*(q+nominal)]:
                for transition in [.25,.8]:
                    fraction=np.minimum(knots/transition,1.)[:,None]
                    nodes=norm(previous[None,:]+fraction*(goal-previous)[None,:])
                    proposals[key].append(nodes)
        evaluate_proposals(proposals)
        for generation,sigma in enumerate([.30,.20,.12,.07,.04],start=1):
            if time.monotonic()>deadline-120:break
            proposals={key:[] for key in records}
            for key,record in records.items():
                knots=knots_for(record);elite=sorted(candidates[key],key=lambda r:(r['rank'],r['proposal_index']))[:4]
                for index in range(24):
                    perturb=rng[key].normal(0,.3*sigma,(len(knots),18))
                    extension=rng[key].normal(0,sigma,(len(knots),1))+rng[key].normal(0,.4*sigma,(len(knots),4))
                    perturb[:,8:12]+=extension;perturb[:,4:8]-=.5*extension
                    perturb[:,:4]+=rng[key].normal(0,.5*sigma,(len(knots),1))
                    perturb[:,12:]+=rng[key].normal(0,sigma,(len(knots),6))
                    perturb[1:]=.5*perturb[1:]+.5*perturb[:-1]
                    if index%2==0:perturb[knots<.25]=0.
                    nodes=np.clip(np.asarray(elite[index%len(elite)]['nodes'])+2*perturb/span,-1,1)
                    proposals[key].append(nodes)
            evaluate_proposals(proposals)
        # Every candidate already used the fine model; do not spend the
        # remaining budget repeating the same deterministic simulation.
        fine={key:sorted(candidates[key],key=lambda r:(r['rank'],r['proposal_index']))[:3] for key in records}
    summary=[]
    for key in records:
        chosen=min(fine[key],key=lambda r:r['rank']);count=prefixes[key];original=sources[key]
        prefix=original['fine']['trajectory'][:count]
        suffix=[dict(row,step=count+row['step']) for row in chosen['trajectory']]
        combined=dict(chosen,trajectory=prefix+suffix,steps=count+chosen['steps'],
            required_steps=800-original['initial_actual_state']['tick'],
            command_representation='full PCHIP recovery from recorded native release',
            nodes_describe_suffix_only=False,prefix_steps=0,first_command_free=True)
        failure=chosen['failure']
        if failure is not None:combined['failure']=dict(failure,step=count+failure['step'])
        if combined['model_path_complete'] and len(combined['trajectory'])!=combined['required_steps']:
            raise ValueError('Incomplete combined candidate')
        result=dict(state_key=key,initial_actual_state=original['initial_actual_state'],
            model_split_state=records[key],prefix_steps=count,fine=combined,
            fine_suffix_alternatives=[{k:v for k,v in row.items() if k not in ['trajectory','terminal']} for row in fine[key]],
            native_success=False,valid_expert_labels=False,full_episode_certified=False,
            prefix_is_model_not_native=False,model_horizon_reaches_original_episode_end=True)
        write('state_'+key.replace(':','_')+'.json',result)
        summary.append(dict(state_key=key,steps=combined['steps'],model_path_complete=combined['model_path_complete'],
            terminal_stable_s=combined['terminal_stable_s'],failure=combined['failure']))
    for f,h in pins.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed input '+f)
    result=dict(state='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH',method='released_prefix_full18_fine_recovery_search',all_candidates_substeps=160,
        completed_candidates=completed,results=summary,wall_seconds=time.monotonic()-started,
        scheduling='round_robin_candidates_across_states',equal_cpu_time_guaranteed=False,
        per_state_compute={key:dict(completed_candidates=len(rows),
            budget_limited=sum((r.get('failure') or {}).get('reason')=='compute_budget' for r in rows),
            worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),
            worker_wall_seconds=sum(r['worker_wall_seconds'] for r in rows),
            maximum_queue_wait_seconds=max((r['queue_wait_seconds'] for r in rows),default=0.)) for key,rows in candidates.items()},
        native_success=False,valid_expert_labels=False)
    write('result.json',result);write('status.json',result);print(json.dumps(plain(result)))


if __name__=='__main__':main()
