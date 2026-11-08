"""Full-tail predictions around a twice-executed successful recovery reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.position_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.executed_nominal_recovery_v1 import executed_reference
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    w=_WORKER;c=w['controller'];i=job['tick']-280;t=job['trace']
    query=dict(q=t['q'][i],v=t['v'][i],contacts=t['contacts'][i],target=t['previous_target_rad'][i])
    def run():
        policy=executed_reference(c.qp.model,c.qp.interface,t,job['gains'])
        return evaluate_suffix(c,state_from_row(query),policy,w['spheres'],
            HipMountPathConstraintV1(w['geometry'],job['geometry'],margin_rad=.02),w['protocol'],
            control_tick=job['tick'],substeps=160,deadline=job['deadline'])
    result=run();exact=False
    if result['accepted']:
        repeat=run();exact=repeat['accepted'] and len(repeat['trajectory'])==len(result['trajectory']) and all(
            np.array_equal(x[k],y[k]) for x,y in zip(result['trajectory'],repeat['trajectory']) for k in ['q','v','target','contacts'])
    return dict(tick=job['tick'],result=result,independent_replay_exact=exact)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','run','bundle','native-audit','repeat-audit','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh reference check required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    setup=read(a.model_plan);m=read(a.bundle/'manifest.json');audit=read(a.native_audit);repeat=read(a.repeat_audit)
    if (read(a.run/'job.json')['status']!='SUCCEEDED' or not audit['passed_dispatch_integrity']
            or not audit['selected_case_outcome']['recovery']):raise ValueError('Actually recovered source required')
    comparison=repeat['comparisons'][0]
    if comparison['case']!=m['case'] or comparison['maximum_public_error']!=0 or comparison['maximum_action_error']!=0:
        raise ValueError('Repeated case reference differs')
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    if context[0]!=m['configuration']:raise ValueError('Original physical interface differs')
    initialize(*context);w=_WORKER;c=w['controller']
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    ticks=np.array([r['tick'] for r in read(a.run/'sequence_native_result.json')['records']],int)
    if not np.array_equal(ticks,np.arange(280,800)):raise ValueError('Full native command record required')
    policy=executed_reference(c.qp.model,c.qp.interface,trace,m['tracking_gains'])
    max_requested=0.;max_encoded=0.;encoded_exact=True
    for i in range(520):
        state=state_from_row(dict(q=trace['q'][i],v=trace['v'][i],contacts=trace['contacts'][i],target=trace['previous_target_rad'][i]))
        target,_=policy.action(80+i,state,state.previous_command_target)
        max_requested=max(max_requested,float(np.max(abs(target-trace['requested_target_rad'][i]))))
        encoded=(2*(target-c.qp.interface.limits[:,0])/np.diff(c.qp.interface.limits).ravel()-1).astype(np.float32)
        encoded_exact=encoded_exact and np.array_equal(encoded,trace['final_action'][i])
        max_encoded=max(max_encoded,float(np.max(abs(encoded-trace['final_action'][i]))))
    if not encoded_exact:raise ValueError('Zero-error reference does not reproduce actual dispatch')
    pins=dict(audit['source_sha256'])
    for path in [a.model_plan,a.bundle/'manifest.json',a.native_audit,a.repeat_audit,Path(__file__).resolve()]:pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());digest=sha(path)
            if path in pins and pins[path]!=digest:raise ValueError('Pinned source changed '+path)
            pins[path]=digest
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Source changed '+path)
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    write('inputs.json',dict(configuration={k:str(v) for k,v in vars(a).items()},source_sha256=pins,max_seconds=600,workers=2,
        interface_check=dict(steps=520,encoded_exact=encoded_exact,maximum_requested_error_rad=max_requested,maximum_encoded_error=max_encoded)))
    began=time.monotonic();deadline=began+600
    jobs=[dict(tick=t,trace=trace,gains=m['tracking_gains'],geometry=geometry,deadline=deadline) for t in [280,400,500]]
    try:
        write('status.json',dict(state='PREDICTING_EXECUTED_NOMINAL_REFERENCE'))
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
            rows=[]
            for row in pool.map(evaluate,jobs):
                rows.append(row);write(f"tick_{row['tick']}.json",row)
                print(json.dumps(dict(tick=row['tick'],accepted=row['result']['accepted'],completed=row['result']['completed_model_steps'],failure=row['result']['failure'])),flush=True)
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Input changed during prediction '+path)
        write('result.json',dict(scope=__doc__,rows=rows,source_sha256=pins,wall_seconds=time.monotonic()-began,
            strict_untouched_isolation_passed=audit['strict_untouched_isolation_passed'],native_execution=False,
            valid_expert_labels=False,physical_constraints_changed=False))
        write('status.json',dict(state='COMPLETED',accepted_ticks=[r['tick'] for r in rows if r['independent_replay_exact']]))
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
