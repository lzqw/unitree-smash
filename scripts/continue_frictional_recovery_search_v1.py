"""Continue a failed structured search without changing its physical checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_frictional_recovery_cem_v1 import init_worker,evaluate,key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','sphere-audit','start-result','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--population',type=int,default=64);p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--seed',type=int,default=882096);a=p.parse_args()
    if a.output.exists() or not 16<=a.population<=128 or not 1<=a.iterations<=12:raise ValueError('Bounded new search required')
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    start=json.loads(a.start_result.read_text())
    for f,digest in start['source_sha256'].items():
        if sha(f)!=digest:raise ValueError('Changed prior search input '+f)
    if str((a.bundle/'manifest.json').resolve()) not in start['source_sha256'] or str(a.sphere_audit.resolve()) not in start['source_sha256']:
        raise ValueError('Different bundle or sphere evidence')
    hashes=dict(start['source_sha256']);hashes[str(a.start_result.resolve())]=sha(a.start_result)
    hashes[str(Path(__file__).resolve())]=sha(Path(__file__))
    mean=np.asarray(start['knots'],float);std=np.full_like(mean,.02);best_knots=mean.copy()
    rng=np.random.default_rng(a.seed);history=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
            initializer=init_worker,initargs=(str(a.bundle),str(a.sphere_audit),start['case'])) as pool:
        baseline=pool.submit(evaluate,mean).result();best=baseline
        if key(baseline)!=key(start['final']):raise ValueError('Cannot reproduce previous best')
        for iteration in range(a.iterations):
            candidates=mean+rng.normal(size=(a.population,3,18))*std;candidates[0]=best_knots
            values=list(pool.map(evaluate,candidates));order=sorted(range(len(values)),key=lambda i:key(values[i]))
            elite=candidates[order[:max(6,a.population//6)]]
            mean=.2*mean+.8*elite.mean(0);std=np.maximum(.001,.3*std+.7*elite.std(0))
            if key(values[order[0]])<key(best):best=values[order[0]];best_knots=candidates[order[0]].copy()
            row=dict(iteration=iteration,best=best,accepted=sum(v['accepted'] for v in values),solver_failures=sum(not v['solved'] for v in values));history.append(row)
            print(json.dumps(row),flush=True)
            partial=dict(history=history,best_knots=best_knots.tolist(),mean=mean.tolist(),std=std.tolist(),
                random_state=rng.bit_generator.state,source_sha256=hashes,native_execution=False,valid_teacher=False)
            temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(partial,indent=2,allow_nan=False)+'\n');temp.replace(a.output.with_suffix('.partial.json'))
        final=pool.submit(evaluate,best_knots,True).result()
    if key(final)!=key(best):raise ValueError('Independent replay differs')
    for f,digest in hashes.items():
        if sha(f)!=digest:raise ValueError('Search input changed '+f)
    result=dict(scope=__doc__,native_execution=False,valid_teacher=False,case=start['case'],seed=a.seed,
        population=a.population,iterations=a.iterations,baseline=baseline,history=history,final=final,
        knots=best_knots.tolist(),wall_seconds=time.perf_counter()-started,source_sha256=hashes,
        limitation='Model candidate only; requires longer feedback execution and native verification')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
