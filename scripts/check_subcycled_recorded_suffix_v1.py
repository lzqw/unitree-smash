"""Full recorded suffix propagation using refined contact integration.

The original diagnostic retains every geometry, command, and terminal gate.
Only its step function is supplied by this explicit private-worker adapter.
This is a model comparison, never a new native rollout or a teacher label.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.subcycled_foot_pair_step_v1 import subcycled_foot_pair_step
from scripts.audit_joint_sequence_proposal_v1 import plain
import scripts.check_pair_recorded_suffix_v1 as original


def evaluate(job):
    # A spawned private process owns this adapter. No native controller or
    # original module on disk is modified, and its validation logic is reused.
    step=partial(subcycled_foot_pair_step,subdivisions=8,
        endpoint_velocity=job['variant']=='refined_endpoint')
    old=original.foot_pair_velocity_step
    try:
        original.foot_pair_velocity_step=step
        result=original.evaluate(dict(job,mode='pair_step'))
    finally:
        original.foot_pair_velocity_step=old
    result['mode']=job['variant']
    result['integration_subdivisions']=8
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','run','comparison','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--start-tick',type=int,choices=[280,400],required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    comparison=read(a.comparison)
    if comparison['interval']!='control' or any(r['completed']!=r['intervals'] for r in comparison['summaries'].values()):
        raise ValueError('Complete control-interval comparison required')
    pins=dict(comparison['source_sha256'])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed evidence '+f)
    source=read(read(a.plan)['source']);*_,geometry,context=source_context(source['configuration']['source'])
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    if [r['tick'] for r in read(a.run/'sequence_native_result.json')['records']]!=list(range(280,800)):
        raise ValueError('Full actual 520-command remainder required')
    material=read(a.run/'native_admission.json')['material_admission']['feet'];mu=material[0]['material']['dynamic']
    if any(m['material']['dynamic']!=mu or m['material']['combine']!='average' for m in material):
        raise ValueError('Unexpected physical friction')
    for f in [a.plan,a.comparison,Path(__file__),Path(original.__file__),
            a.run/'sequence_native_trace.npz',a.run/'sequence_native_result.json',a.run/'native_admission.json']:
        pins[str(f.resolve())]=sha(f)
    a.output.mkdir(parents=True)
    def write(name,obj):
        path=a.output/name;temp=path.with_suffix('.tmp');temp.write_text(json.dumps(plain(obj),indent=2,allow_nan=False)+'\n');temp.replace(path)
    write('inputs.json',dict(source_sha256=pins,maximum_seconds=600,workers=2,start_tick=a.start_tick))
    write('status.json',dict(state='RUNNING'))
    deadline=time.monotonic()+600
    jobs=[dict(variant=v,tick=a.start_tick,trace=trace,geometry=geometry,friction=mu,deadline=deadline)
        for v in ['refined','refined_endpoint']]
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
            rows=list(pool.map(evaluate,jobs))
        for f,h in pins.items():
            if sha(f)!=h:raise ValueError('Source changed during propagation '+f)
        write('result.json',dict(scope=__doc__,rows=rows,source_sha256=pins,native_execution=False,
            model_promoted=False,valid_expert_labels=False))
        write('status.json',dict(state='COMPLETED'))
        print(json.dumps([{k:v for k,v in r.items() if k not in ['trajectory','terminal','terminal_violations']} for r in rows]))
    except BaseException as e:
        write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
