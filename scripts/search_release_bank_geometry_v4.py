"""Batch full-command trajectory search from actual causal release states.

Uses geometry-based landing seeds, coupled hip/mount response, and the same objective for every case. The recorded failed controller
commands are only one offline warm start, never positive expert labels. No RL
or native teleport. Every candidate plans all 18 position commands for 600 ms;
successful model prefixes are not native stable landings.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.receding_full_command_search_v1 import initialize_court,warm_knots
from scripts.full_command_self_contact_rollout_v3 import rollout
from scripts.landing_geometry_seeds_v1 import landing_geometry_seeds
from scripts.audit_joint_sequence_proposal_v1 import plain


def search_case(entry,recorded,audit,deadline,nominal):
    start=time.monotonic();query=entry['query'];c=_WORKER['controller'];limits=c.qp.interface.limits
    state=state_from_row(query);previous=np.asarray(query['target']);q=state.joint_position;v=state.joint_velocity
    def normalized(targets):return np.clip(2*(targets-limits[:,0])/np.diff(limits).ravel()-1,-1,1)
    geometry_seeds=landing_geometry_seeds(c.qp.model,state,limits,nominal)
    seeds=[('failed_native_warm_start',warm_knots(recorded,previous,limits))]
    seeds += [(label,nodes) for label,nodes,metadata in geometry_seeds]
    coarse=[]
    for label,nodes in seeds:
        if time.monotonic()>=deadline:break
        r=rollout(nodes,query,audit,16,deadline);r['seed']=label;coarse.append(r)
    if not coarse:return dict(case=entry['case'],status='budget_before_case',native_execution=False)
    coarse.sort(key=lambda r:r['rank']);mean=np.asarray(coarse[0]['nodes']);rng=np.random.default_rng(884001+entry['case'])
    for generation in range(2):
        if time.monotonic()>=deadline:break
        sampled=np.clip(rng.normal(mean,.12 if generation==0 else .06,(6,5,18)),-1,1)
        for nodes in sampled:
            if time.monotonic()>=deadline:break
            r=rollout(nodes,query,audit,16,deadline);r['seed']='sequence_search';coarse.append(r)
        coarse.sort(key=lambda r:r['rank']);mean=np.asarray([r['nodes'] for r in coarse[:3]]).mean(0)
    best=coarse[0]
    # Fine independent dynamics replay uses the same constraints and command
    # sequence. It may disagree with coarse results; never promote coarse-only.
    fine=rollout(best['nodes'],query,audit,160,deadline) if time.monotonic()<deadline else None
    return dict(case=entry['case'],source_control_tick=entry['control_tick'],status='model_search_finished',
        coarse_candidates=[{k:v for k,v in r.items() if k!='trajectory'} for r in coarse],
        seed_metadata=[dict(label=label,**metadata) for label,nodes,metadata in geometry_seeds],
        fine=fine,wall_seconds=time.monotonic()-start,native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['bank','run','model-inputs','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1800.)
    p.add_argument('--workers',type=int,choices=range(1,9),default=8)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists() or not 120<=a.max_seconds<=3600:raise ValueError('Fresh bounded experiment required')
    bank=read(a.bank);x=read(a.model_inputs);s=read(read(x['configuration']['model_plan'])['source'])
    *_,audit,context=source_context(s['configuration']['source']);initialize_court(*context)
    with np.load(a.run/'pure_control_commands.npz') as z:
        lanes=z['lane'];targets=z['final']
    nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    pins=dict(bank['source_sha256'])
    for f in [a.bank,a.model_inputs,Path(__file__),ROOT/"configs/go2_piper_hip_mount_boundaries_v1.json",a.run/"native_command_inputs.json"]:pins[str(f.resolve())]=sha(f)
    for name,module in list(sys.modules.items()):
        if name.startswith(('scripts.','quadsmash.physics.court')) and getattr(module,'__file__',None):
            f=str(Path(module.__file__).resolve());h=sha(f)
            if f in pins and pins[f]!=h:raise ValueError('Changed dependency '+f)
            pins[f]=h
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    a.output.mkdir(parents=True)
    def write(name,data):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(source_sha256=pins,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},cases=len(bank['queries']),
        identical_controller_objective_for_all_cases=True,model_only=True,training=False,
        hip_mount_collision_response=True,geometric_landing_seeds=True,software_arm_command_slew=None,max_seconds=a.max_seconds))
    start=time.monotonic();deadline=start+a.max_seconds;results=[]
    write('status.json',dict(state='RUNNING_RELEASE_BANK_MODEL_SEARCH',completed=0,total=len(bank['queries'])))
    try:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize_court,initargs=context) as pool:
            futures=[pool.submit(search_case,e,targets[lanes==e['case']][:60],audit,deadline,nominal) for e in bank['queries']]
            for future in as_completed(futures):
                r=future.result();results.append(r);write('case_%03d.json'%r['case'],r)
                write('status.json',dict(state='RUNNING_RELEASE_BANK_MODEL_SEARCH',completed=len(results),total=len(futures),
                    fine_full_horizon=sum(bool(v.get('fine') and v['fine']['horizon_model_feasible']) for v in results)))
                if len(results)%10==0:print(json.dumps(read(a.output/'status.json')),flush=True)
        for f,h in pins.items():
            if sha(f)!=h:raise ValueError('Changed source during run '+f)
        summary=dict(state='COMPLETED_MODEL_SEARCH',cases=len(results),wall_seconds=time.monotonic()-start,
            fine_full_horizon_cases=[r['case'] for r in results if r.get('fine') and r['fine']['horizon_model_feasible']],
            budget_unprocessed_cases=[r['case'] for r in results if r['status']=='budget_before_case'],
            native_execution=False,valid_expert_labels=False,full_episode_certified=False)
        write('result.json',summary);write('status.json',summary);print(json.dumps(summary))
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
