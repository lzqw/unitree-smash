"""Extend a dense fine-model prefix into complete recovery using fine dynamics.

The split state is MODEL-PREDICTED, not a native reset or a new observation.
Preserve the checked prefix exactly, optimize its remaining full18 commands,
and reconstruct the complete candidate from the original actual initial state.
No future native commands are used. Full feedback/native checks remain required.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.search_current_native_recovery_v4 import initialize,rollout,knots_for,WORK
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['search','configuration','fixture-run','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prefixes',nargs='+',required=True,help='case:actual_tick:model_prefix_steps')
    p.add_argument('--max-seconds',type=int,default=2400)
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    if not 600<=a.max_seconds<=3600:raise ValueError('Bounded fine search duration required')
    if a.output.exists() or read(a.search/'result.json')['state']!='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH':
        raise ValueError('Fresh output and completed source planning required')
    for f,h in read(a.search/'inputs.json')['source_sha256'].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed source '+f)
    parent=read(a.search/'inputs.json')
    if not parent.get('knots_s'):raise ValueError('Explicit source knot times required')
    cfg=read(a.configuration)
    nominal=read(a.fixture_run/'native_command_inputs.json')['default_joint_position_rad']
    records={};sources={};prefixes={};files=[Path(__file__),a.configuration,
        a.fixture_run/'native_command_inputs.json',a.search/'inputs.json',a.search/'result.json',Path(cfg['model_reference_library'])]
    for item in a.prefixes:
        case,tick,count=map(int,item.split(':'));key=str(case)+':'+str(tick)
        path=a.search/('state_'+key.replace(':','_')+'.json');source=read(path)
        if key in records or source['native_success'] or source['valid_expert_labels']:
            raise ValueError('Distinct model-only candidate required')
        fine=source['fine'];initial=source['initial_actual_state'];trace=fine['trajectory']
        source_knots=np.asarray(parent['knots_s'][key],float)
        if len(source_knots)!=len(fine['nodes']) or np.any(np.diff(source_knots)<=0):
            raise ValueError('Source node count and declared knot times disagree')
        if fine['substeps']!=160 or not 20<=count<len(trace) or 800-tick-count<100:
            raise ValueError('Checked fine prefix and at least one second remaining required')
        if initial['case']!=case or initial['tick']!=tick:raise ValueError('Wrong source state')
        if any(r['step']!=i for i,r in enumerate(trace[:count])):raise ValueError('Discontinuous prefix')
        last=trace[count-1]
        records[key]=dict(q=last['q'],v=last['v'],previous=last['target'],ground_force=last['ground_force'],
            phase=last['phase'],case=case,tick=tick+count,
            time_s=float(np.float32(tick+count)*np.float32(.01)),state_origin='model_prefix_not_native')
        prefixes[key]=count;sources[key]=source;files.append(path)
    initialize(cfg,nominal,records);c=WORK['c'];limits=c.interface.limits;span=np.diff(limits).ravel()
    norm=lambda q:np.clip(2*(np.asarray(q)-limits[:,0])/span-1,-1,1)
    a.output.mkdir(parents=True)
    def write(name,value):
        f=a.output/name;t=f.with_suffix('.tmp');t.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');t.replace(f)
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    write('inputs.json',dict(source_sha256=pins,model_split_states=records,prefix_steps=prefixes,
        original_actual_initial_states={k:s['initial_actual_state'] for k,s in sources.items()},
        native_state_teleport=False,future_native_commands_used=False,training=False,
        workers=4,max_seconds=a.max_seconds,source_knots_s=parent['knots_s'],all_candidates_substeps=160,physical_limits_changed=False,
        prefix_checked_at_substeps=160,prefix_preservation='exact previously checked commands and predicted states'))
    started=time.monotonic();deadline=started+a.max_seconds;candidates={k:[] for k in records}
    rng={k:np.random.default_rng(887000+i) for i,k in enumerate(records)};completed=0
    write('status.json',dict(state='SEARCHING_FINE_PREFIX_CONTINUATIONS',pid=os.getpid(),completed_candidates=0))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cfg,nominal,records)) as pool:
        jobs=[]
        for key,record in records.items():
            knots=knots_for(record);source=sources[key]
            old=PchipInterpolator(np.asarray(parent['knots_s'][key]),np.asarray(source['fine']['nodes']),axis=0)
            original_tail=np.clip(old(knots+prefixes[key]*.01),-1,1)
            jobs.append((key,pool.submit(rollout,key,original_tail,160,deadline)))
            q=np.asarray(record['q'])[c.model.q_indices];previous=np.asarray(record['previous'])
            for goal in [q,previous,np.asarray(nominal),.5*(q+nominal)]:
                for transition in [.25,.8]:
                    fraction=np.minimum(knots/transition,1.)[:,None]
                    nodes=norm(previous[None,:]+fraction*(goal-previous)[None,:])
                    jobs.append((key,pool.submit(rollout,key,nodes,160,deadline)))
        for key,job in jobs:candidates[key].append(job.result());completed+=1
        for generation,sigma in enumerate([.30,.20,.12,.07,.04]):
            if time.monotonic()>deadline-120:break
            jobs=[]
            for key,record in records.items():
                knots=knots_for(record);elite=sorted(candidates[key],key=lambda r:r['rank'])[:4]
                for index in range(24):
                    perturb=rng[key].normal(0,.3*sigma,(len(knots),18))
                    extension=rng[key].normal(0,sigma,(len(knots),1))+rng[key].normal(0,.4*sigma,(len(knots),4))
                    perturb[:,8:12]+=extension;perturb[:,4:8]-=.5*extension
                    perturb[:,:4]+=rng[key].normal(0,.5*sigma,(len(knots),1))
                    perturb[:,12:]+=rng[key].normal(0,sigma,(len(knots),6))
                    perturb[1:]=.5*perturb[1:]+.5*perturb[:-1]
                    if index%2==0:perturb[knots<.25]=0.
                    nodes=np.clip(np.asarray(elite[index%len(elite)]['nodes'])+2*perturb/span,-1,1)
                    jobs.append((key,pool.submit(rollout,key,nodes,160,deadline)))
            for key,job in jobs:candidates[key].append(job.result());completed+=1
            write('status.json',dict(state='SEARCHING_FINE_PREFIX_CONTINUATIONS',pid=os.getpid(),
                generation=generation+1,completed_candidates=completed,elapsed_s=time.monotonic()-started))
        # Every candidate already used the fine model; do not spend the
        # remaining budget repeating the same deterministic simulation.
        fine={key:sorted(candidates[key],key=lambda r:r['rank'])[:3] for key in records}
    summary=[]
    for key in records:
        chosen=min(fine[key],key=lambda r:r['rank']);count=prefixes[key];original=sources[key]
        prefix=original['fine']['trajectory'][:count]
        suffix=[dict(row,step=count+row['step']) for row in chosen['trajectory']]
        combined=dict(chosen,trajectory=prefix+suffix,steps=count+chosen['steps'],
            required_steps=800-original['initial_actual_state']['tick'],
            command_representation='fixed checked prefix plus PCHIP suffix',
            nodes_describe_suffix_only=True,prefix_steps=count)
        failure=chosen['failure']
        if failure is not None:combined['failure']=dict(failure,step=count+failure['step'])
        if combined['model_path_complete'] and len(combined['trajectory'])!=combined['required_steps']:
            raise ValueError('Incomplete combined candidate')
        result=dict(state_key=key,initial_actual_state=original['initial_actual_state'],
            model_split_state=records[key],prefix_steps=count,fine=combined,
            fine_suffix_alternatives=[{k:v for k,v in row.items() if k not in ['trajectory','terminal']} for row in fine[key]],
            native_success=False,valid_expert_labels=False,full_episode_certified=False,
            prefix_is_model_not_native=True,model_horizon_reaches_original_episode_end=True)
        write('state_'+key.replace(':','_')+'.json',result)
        summary.append(dict(state_key=key,steps=combined['steps'],model_path_complete=combined['model_path_complete'],
            terminal_stable_s=combined['terminal_stable_s'],failure=combined['failure']))
    for f,h in pins.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('Changed input '+f)
    result=dict(state='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH',method='dense_checked_prefix_and_fine_only_suffix_search',all_candidates_substeps=160,
        completed_candidates=completed,results=summary,wall_seconds=time.monotonic()-started,
        native_success=False,valid_expert_labels=False)
    write('result.json',result);write('status.json',result);print(json.dumps(plain(result)))


if __name__=='__main__':main()
