"""Offline full18 recovery optimization with completed model feedback seeds.

Native initial observations are checked directly. Later warm-start commands
come from model feedback, never later native actions. Every scored candidate
runs toward the original 8 s endpoint with the original physical checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.full_recovery_margin_rollout_v1 import rollout, initialize, knots_for, WORK
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['feedback','configuration','run','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--states',nargs='+',required=True)
    p.add_argument('--max-seconds',type=int,default=2400)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if a.output.exists() or not 600<=a.max_seconds<=3600 or len(set(a.states))!=len(a.states):
        raise ValueError('Fresh bounded search with distinct states required')
    parent=read(a.feedback/'inputs.json');done=read(a.feedback/'result.json')
    if done['state']!='COMPLETED_ARCHIVED_REFERENCE_RECHECK' or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Completed baseline model feedback and native initial history required')
    pins=dict(parent['source_sha256'])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    if pins.get(str(a.configuration.resolve()))!=sha(a.configuration):raise ValueError('Keep baseline physics configuration')
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    records={};sources={};files=[Path(__file__),a.configuration,a.feedback/'inputs.json',a.feedback/'result.json',a.run/'job.json',a.run/'pure_control_commands.npz',a.run/'native_command_inputs.json']
    with np.load(a.run/'pure_control_commands.npz') as z:
        for key in a.states:
            path=a.feedback/('state_'+key.replace(':','_')+'.json');source=read(path);record=source['initial_actual_state'];trace=source['trajectory']
            if source['failure'] is not None or not source['full_model_path'] or source['final_model_stable_s']<.8 or source['native_success'] or source['valid_expert_labels']:
                raise ValueError('A completed stable model seed is required')
            if source['state_key']!=key or len(trace)!=800-record['tick'] or any(row['tick']!=record['tick']+i for i,row in enumerate(trace)):
                raise ValueError('Incorrect reference clock')
            ids=np.flatnonzero((z['lane']==record['case'])&(z['tick']==record['tick']))
            if len(ids)!=1 or any(not np.array_equal(z[k][ids[0]],record[k]) for k in ['q','v','previous','ground_force']):
                raise ValueError('Actual initial observation mismatch')
            commands=np.asarray([row['target'] for row in trace]);warmstart=np.vstack([commands,commands[-1]])
            records[key]=dict(record,warmstart_targets=warmstart,state_origin='recorded_native_initial_state')
            sources[key]=source;files.append(path)
    initialize(cfg,nominal,records);c=WORK['c'];limits=c.interface.limits;span=np.diff(limits).ravel()
    normalize=lambda values:np.clip(2*(values-limits[:,0])/span-1,-1,1)
    seeds={key:normalize(record['warmstart_targets'][np.rint(knots_for(record)/.01).astype(int)]) for key,record in records.items()}
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    pins.update({str(f.resolve()):sha(f) for f in files})
    a.output.mkdir(parents=True)
    write(a.output/'inputs.json',dict(source_sha256=pins,states=a.states,records=records,workers=4,max_seconds=a.max_seconds,
        physical_substeps=160,surface_sample_period_s=.01,margin_preference_m=.01,
        objective_order=['complete physical path and original stability','sampled nonfoot margin','integrated margin penalty','motion cost'],
        first_command_free=True,full18_authority=True,native_teleport=False,future_native_commands_used=False,
        training=False,acceptance_changed=False,physical_limits_changed=False))
    started=time.monotonic();deadline=started+a.max_seconds;results={key:[] for key in records};generation=0
    rng={key:np.random.default_rng(889000+i) for i,key in enumerate(records)}
    def progress(state):
        write(a.output/'status.json',dict(state=state,pid=os.getpid(),generation=generation,elapsed_s=time.monotonic()-started,
            per_state={key:dict(candidates=len(rows),complete_stable=sum(r['complete_stable_model'] for r in rows),
                best_margin_m=min(rows,key=lambda r:r['rank'])['sampled_minimum_nonfoot_m'] if rows else None,
                budget_limited=sum((r['failure'] or {}).get('reason')=='compute_budget' for r in rows)) for key,rows in results.items()},
            native_success=False,valid_expert_labels=False))
    progress('RUNNING_FULL_RECOVERY_MARGIN_SEARCH')
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(cfg,nominal,records)) as pool:
        def evaluate(proposals):
            jobs={}
            for index in range(max(map(len,proposals.values()))):
                for key,nodes in proposals.items():
                    if index<len(nodes):jobs[pool.submit(rollout,key,nodes[index],deadline)]=key
            for future in as_completed(jobs):
                key=jobs[future];row=future.result();results[key].append(row)
                selected=min(results[key],key=lambda r:r['rank'])
                write(a.output/('state_'+key.replace(':','_')+'.json'),dict(state_key=key,initial_actual_state=sources[key]['initial_actual_state'],
                    fine=selected,model_split_state=records[key],native_success=False,valid_expert_labels=False))
                progress('RUNNING_FULL_RECOVERY_MARGIN_SEARCH')
        evaluate({key:[nodes] for key,nodes in seeds.items()})
        seed_checks=[]
        for key,rows in results.items():
            row=rows[0];source=sources[key]
            if not row['complete_stable_model']:raise ValueError('Exact model command seed did not complete stable recovery '+key)
            command_error=float(np.max(abs(np.asarray([x['target'] for x in row['trajectory']])-records[key]['warmstart_targets'][:-1])))
            q_error=float(np.max(abs(np.asarray([x['q'] for x in row['trajectory']])-np.asarray([x['q'] for x in source['trajectory']]))))
            v_error=float(np.max(abs(np.asarray([x['v'] for x in row['trajectory']])-np.asarray([x['v'] for x in source['trajectory']]))))
            if command_error>1e-12 or max(q_error,v_error)>1e-5:raise ValueError('Warm-start model replay not reproduced '+key)
            seed_checks.append(dict(state_key=key,command_error_rad=command_error,q_error=q_error,v_error=v_error,minimum_nonfoot_m=row['sampled_minimum_nonfoot_m']))
        write(a.output/'seed_check.json',dict(passed=True,checks=seed_checks,native_execution=False))
        for generation,sigma in enumerate([.025,.06,.015,.008],start=1):
            if time.monotonic()>deadline-180:break
            proposals={key:[] for key in records}
            for key,rows in results.items():
                elite=sorted(rows,key=lambda r:r['rank'])[:3];knots=knots_for(records[key])
                for i in range(8):
                    noise=rng[key].normal(0.,sigma,(len(knots),18))
                    if i%2==0:
                        extension=rng[key].normal(0.,sigma,(len(knots),4));noise[:,8:12]+=extension;noise[:,4:8]-=.5*extension
                    noise[1:]=.5*(noise[1:]+noise[:-1])
                    # Sampling focuses on impact without freezing any command
                    # or treating this standard deviation as an action limit.
                    noise[knots>1.2]*=.2
                    proposals[key].append(np.clip(elite[i%len(elite)]['nodes']+2*noise/span,-1.,1.))
            evaluate(proposals)
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed optimization input '+f)
    summary=[]
    for key,rows in results.items():
        best=min(rows,key=lambda r:r['rank']);base=rows[0]
        summary.append(dict(state_key=key,candidates=len(rows),complete_stable=sum(r['complete_stable_model'] for r in rows),
            baseline_minimum_nonfoot_m=base['sampled_minimum_nonfoot_m'],best_minimum_nonfoot_m=best['sampled_minimum_nonfoot_m'],
            complete_stable_model=best['complete_stable_model'],terminal_stable_s=best['terminal_stable_s'],
            improved=best['complete_stable_model'] and best['sampled_minimum_nonfoot_m']>base['sampled_minimum_nonfoot_m']+1e-6))
    final=dict(state='COMPLETED_FULL_RECOVERY_MARGIN_SEARCH',results=summary,elapsed_s=time.monotonic()-started,
               native_success=False,valid_expert_labels=False)
    write(a.output/'result.json',final);progress(final['state']);print(json.dumps(final))


if __name__=='__main__':main()
