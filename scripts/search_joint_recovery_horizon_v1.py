"""Jointly optimize 0.8 s instead of accepting a nonviable 0.4 s endpoint.

First six commands retain the verified prefix; subsequent commands may change
before and after the old endpoint. Every microstep uses the unchanged full
frictional predictor and original physical acceptance checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.search_frictional_recovery_cem_v1 as core
from scripts.extend_frictional_recovery_v1 import set_state
from scripts.sphere_recontact_reference_v1 import propose_recontact
from scripts.frictional_static_reference_v1 import sphere_static_reference
from scripts.audit_ramped_gravity_continuation_v1 import rate_admitted_seed

TIMES=np.array([0.,2.,5.,12.,25.,39.,55.,79.])


def evaluate_task(task):
    c,state,previous,_,spheres,geometry=core.CONTEXT
    seed=np.asarray(task['seed'],float)
    free=np.asarray(task['knots'],float)
    if seed.shape!=(80,18) or free.shape!=(5,18):
        raise ValueError('Require complete declared 0.8 second sequence')
    knots=np.vstack([np.zeros((3,18)),free])
    dense=np.column_stack([np.interp(np.arange(80),TIMES,knots[:,j]) for j in range(18)])
    core.CONTEXT=(c,state,previous,seed,spheres,geometry)
    return core.evaluate(dense,task.get('save',False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','prefix-result','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--population',type=int,default=64)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--seed',type=int,default=882099)
    a=p.parse_args()
    if a.output.exists() or not 16<=a.population<=128 or not 1<=a.iterations<=16:
        raise ValueError('Bounded new search required')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    prefix=json.loads(a.prefix_result.read_text())
    if not prefix['final']['accepted']:
        raise ValueError('Require verified short prefix')
    hashes=dict(prefix['source_sha256'])
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Changed prefix input '+f)
    if str((a.bundle/'manifest.json').resolve()) not in hashes or str(a.sphere_audit.resolve()) not in hashes:
        raise ValueError('Different bundle or sphere evidence')
    m=json.loads((a.bundle/'manifest.json').read_text())
    if sha(a.bundle/m['bundle_name'])!=m['bundle_sha256']:raise ValueError('Changed original archive')
    core.init_worker(str(a.bundle),str(a.sphere_audit),prefix['case'])
    c,initial,previous,_,spheres,geometry=core.CONTEXT
    commands=np.asarray(prefix['final']['commands']);last=prefix['final']['states'][-1]
    if commands.shape!=(40,18):raise ValueError('Require forty-command prefix')
    endpoint=set_state(initial,c.qp.model,last['q'],last['v'],last['contacts'],commands[-1])
    seeds=[('held_tail',np.vstack([commands,np.repeat(commands[-1][None],40,axis=0)]))]
    gravity=sphere_static_reference(c,endpoint,spheres)
    if gravity['solved']:
        seeds.append(('surface_gravity_tail',np.vstack([commands,rate_admitted_seed(c,endpoint,gravity['target'],40)])))
    recontact=propose_recontact(c,endpoint,spheres)
    for row in recontact['seeds']:
        seeds.append((row['name'],np.vstack([commands,row['commands']])))
    with np.load(a.bundle/m['bundle_name']) as z:
        tail=z[f"case_{prefix['case']:03d}_commands_rad"][40:80].copy()
    if tail.shape==(40,18):seeds.append(('old_nominal_tail',np.vstack([commands,tail])))
    files=[Path(__file__),a.prefix_result,c.qp.model.path]
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    for f in files:hashes[str(f.resolve())]=sha(f)
    rng=np.random.default_rng(a.seed);history=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
            initializer=core.init_worker,initargs=(str(a.bundle),str(a.sphere_audit),prefix['case'])) as pool:
        zeros=np.zeros((5,18))
        seed_results=list(pool.map(evaluate_task,[dict(seed=x.tolist(),knots=zeros.tolist()) for _,x in seeds]))
        selected=min(range(len(seeds)),key=lambda i:core.key(seed_results[i]))
        label,seed=seeds[selected];best=seed_results[selected]
        best_knots=zeros.copy();mean=zeros.copy()
        std=np.tile(np.r_[np.full(12,.06),np.full(6,.03)],(5,1));std[0]*=.4
        for iteration in range(a.iterations):
            candidates=mean+rng.normal(size=(a.population,5,18))*std;candidates[0]=best_knots
            values=list(pool.map(evaluate_task,[dict(seed=seed.tolist(),knots=x.tolist()) for x in candidates]))
            order=sorted(range(len(values)),key=lambda i:core.key(values[i]))
            elite=candidates[order[:max(6,a.population//6)]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.0005,.3*std+.7*elite.std(0))
            if core.key(values[order[0]])<core.key(best):best=values[order[0]];best_knots=candidates[order[0]].copy()
            row=dict(iteration=iteration,best=best,accepted=sum(v['accepted'] for v in values),
                solver_failures=sum(not v['solved'] for v in values));history.append(row);print(json.dumps(row),flush=True)
            partial=dict(seed_name=label,seed_commands=seed.tolist(),history=history,best_knots=best_knots.tolist(),
                mean=mean.tolist(),std=std.tolist(),random_state=rng.bit_generator.state,source_sha256=hashes,
                native_execution=False,valid_teacher=False)
            tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(partial,indent=2,allow_nan=False)+'\n')
            tmp.replace(a.output.with_suffix('.partial.json'))
        final=pool.submit(evaluate_task,dict(seed=seed.tolist(),knots=best_knots.tolist(),save=True)).result()
    if core.key(final)!=core.key(best):raise ValueError('Independent replay differs')
    if not np.allclose(np.asarray(final.get('commands',seed))[:6],commands[:6],atol=1e-14,rtol=0):
        raise ValueError('Protected early commands changed')
    for f,h in hashes.items():
        if sha(f)!=h:raise ValueError('Search input changed '+f)
    result=dict(scope=__doc__,case=prefix['case'],horizon_steps=80,seed=a.seed,population=a.population,
        iterations=a.iterations,seed_name=label,seed_results=[dict(name=n,result=r) for (n,_),r in zip(seeds,seed_results)],
        seed_commands=seed.tolist(),history=history,final=final,knots=best_knots.tolist(),
        source_sha256=hashes,wall_seconds=time.perf_counter()-started,native_execution=False,valid_teacher=False,
        limitation='An 0.8 s model plan is not a full episode recovery or a native stable window')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
