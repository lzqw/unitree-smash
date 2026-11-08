"""Bounded full18 sequence refinement using a completed geometry-seeded case.

Diagnostic case selection uses the longest model prefix available at launch,
not future native labels. Every state is an actual recorded release state.
This is offline model optimization, not a native landing result or teacher.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,multiprocessing as mp,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.receding_full_command_search_v1 import initialize_court
from scripts.full_command_self_contact_rollout_v3 import rollout
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['bank','search','model-inputs','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=600.)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists() or not 120<=a.max_seconds<=1200:raise ValueError('Fresh bounded diagnostic required')
    options=[(f,read(f)) for f in sorted(a.search.glob('case_*.json'))]
    options=[(f,r) for f,r in options if r.get('fine') and r['fine']['steps']>=10]
    if not options:raise ValueError('No evaluated sequence prefix available')
    selected,record=min(options,key=lambda pair:pair[1]['fine']['rank'])
    entry=next(e for e in read(a.bank)['queries'] if e['case']==record['case']);query=entry['query']
    x=read(a.model_inputs);s=read(read(x['configuration']['model_plan'])['source'])
    *_,audit,context=source_context(s['configuration']['source']);initialize_court(*context)
    pins={str(f.resolve()):sha(f) for f in [a.bank,selected,a.model_inputs,Path(__file__),ROOT/'configs/go2_piper_hip_mount_boundaries_v1.json']}
    for name,module in list(sys.modules.items()):
        if name.startswith(('scripts.','quadsmash.physics.court')) and getattr(module,'__file__',None):
            f=str(Path(module.__file__).resolve());pins[f]=sha(f)
    a.output.mkdir(parents=True)
    def write(name,data):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(source_sha256=pins,case=entry['case'],query=query,selection='longest completed model prefix, then task cost',
        selection_pool=[dict(file=str(f),case=r['case'],rank=r['fine']['rank']) for f,r in options],
        full18=True,physical_limits_unchanged=True,native_execution=False,training=False,max_seconds=a.max_seconds))
    start=time.monotonic();deadline=start+a.max_seconds;rng=np.random.default_rng(884417)
    best=record['fine'];mean=np.asarray(best['nodes']);std=np.full((5,18),.12);archive=[];history=[]
    write('status.json',dict(state='RUNNING_MODEL_SEQUENCE_REFINEMENT',case=entry['case']))
    try:
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize_court,initargs=context) as pool:
            for generation in range(8):
                if deadline-time.monotonic()<60:break
                noise=rng.normal(size=(48,5,18))
                for knot in range(1,5):noise[:,knot]=.7*noise[:,knot-1]+np.sqrt(1-.7**2)*noise[:,knot]
                nodes=np.clip(mean+std*noise,-1,1);nodes[0]=best['nodes'];nodes[1]=mean
                coarse=[f.result() for f in [pool.submit(rollout,n,query,audit,32,deadline-45) for n in nodes]]
                archive.extend(coarse);archive.sort(key=lambda r:r['rank']);archive=archive[:12]
                elite=np.asarray([r['nodes'] for r in archive[:6]])
                mean=elite.mean(0);std=np.maximum(elite.std(0),.025)
                fine=[f.result() for f in [pool.submit(rollout,r['nodes'],query,audit,160,deadline) for r in archive[:2]]]
                best=min([best,*fine],key=lambda r:r['rank'])
                row=dict(generation=generation,coarse=[{k:v for k,v in r.items() if k!='trajectory'} for r in coarse],fine=fine)
                history.append(row)
                write('partial.json',dict(case=entry['case'],best=best,history=history))
                status=dict(state='RUNNING_MODEL_SEQUENCE_REFINEMENT',case=entry['case'],generation=generation,
                    best_model_steps=best['steps'],full_horizon=best['horizon_model_feasible'],terminal=best['terminal'])
                write('status.json',status);print(json.dumps(status),flush=True)
        for f,h in pins.items():
            if sha(f)!=h:raise ValueError('Changed source during refinement '+f)
        result=dict(state='COMPLETED_MODEL_SEQUENCE_REFINEMENT',case=entry['case'],best=best,history=history,
            wall_seconds=time.monotonic()-start,native_execution=False,valid_expert_labels=False,full_episode_certified=False)
        write('result.json',result);write('status.json',{k:v for k,v in result.items() if k not in ['best','history']})
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
