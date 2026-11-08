"""Paired measured-command predictions with and without coupled position stops."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.predict_position_limited_spheres_v1 import predict_position_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def evaluate(job):
    if time.monotonic()>job['deadline']:return dict(tick=job['tick'],completed=False,error='declared_compute_budget')
    w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    state=state_from_row(job['before']);actual=state_from_row(job['after'])
    calf=model.joint_order.index('RL_calf_joint');results={};states={}
    for label,fn in [('old',predict_speed_limited_spheres),('stops',predict_position_limited_spheres)]:
        try:
            predicted=fn(model,interface,state,job['target'],w['spheres'],geometry=w['geometry'],substeps=160)
            following=predicted['state'];delta=pin.difference(model.model,actual.q,following.q)
            results[label]=dict(completed=True,diagnostics=predicted['diagnostics'],
                root_position_error_m=float(np.linalg.norm(delta[:3])),root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
                joint_position_error_rad=float(np.max(abs(delta[model.v_indices]))),
                joint_velocity_error_rad_s=float(np.max(abs(following.v[model.v_indices]-actual.v[model.v_indices]))),
                calf_position_error_rad=float(abs(following.q[model.q_indices[calf]]-actual.q[model.q_indices[calf]])),
                calf_velocity_error_rad_s=float(abs(following.v[model.v_indices[calf]]-actual.v[model.v_indices[calf]])))
            states[label]=following
        except (ValueError,np.linalg.LinAlgError) as e:results[label]=dict(completed=False,error=repr(e))
    same=None
    if len(states)==2 and results['stops']['diagnostics']['position_limit_active_substeps']==0:
        same=bool(np.array_equal(states['old'].q,states['stops'].q) and np.array_equal(states['old'].v,states['stops'].v))
        if not same:raise ValueError('Inactive stop changed original numerical path')
    return dict(tick=job['tick'],completed=len(states)==2,inactive_exact=same,**results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','run','bundle','native-audit','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh diagnostic required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=read(a.native_audit);manifest=read(a.bundle/'manifest.json');setup=read(a.model_plan)
    if read(a.run/'job.json')['status']!='SUCCEEDED' or not audit['passed_dispatch_integrity']:
        raise ValueError('Completed ground-executed diagnostic required')
    source=read(setup['source']);_,_,_,_,context=source_context(source['configuration']['source'])
    if context[0]!=manifest['configuration']:raise ValueError('Physical configuration differs')
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    native=read(a.run/'sequence_native_result.json');ticks=np.array([r['tick'] for r in native['records']],int)
    if not np.array_equal(ticks,np.arange(280,800)):raise ValueError('Complete actual command history required')
    if np.max(abs(trace['previous_target_rad'][1:]-trace['final_target_rad'][:-1]))>5e-6:raise ValueError('Command history differs')
    pins=dict(audit['source_sha256'])
    for path in [a.model_plan,a.bundle/'manifest.json',a.native_audit,Path(__file__).resolve()]:pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());digest=sha(path)
            if path in pins and pins[path]!=digest:raise ValueError('Pinned source changed '+path)
            pins[path]=digest
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Input changed '+path)
    a.output.mkdir(parents=True)
    def write(name,value):
        path=a.output/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    selected=sorted(set(range(280,330))|set(range(330,799,10))|set(range(408,417)))
    began=time.monotonic();deadline=began+600
    def state(i):return dict(q=trace['q'][i],v=trace['v'][i],contacts=trace['contacts'][i],target=trace['previous_target_rad'][i])
    jobs=[dict(tick=t,before=state(t-280),after=state(t-279),target=trace['final_target_rad'][t-280],deadline=deadline) for t in selected]
    write('inputs.json',dict(configuration={k:str(v) for k,v in vars(a).items()},source_sha256=pins,ticks=selected,max_seconds=600,workers=2))
    write('status.json',dict(state='RUNNING_PAIRED_PREDICTION'))
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:rows=list(pool.map(evaluate,jobs))
        good=[r for r in rows if r['completed']];active=[r for r in good if r['stops']['diagnostics']['position_limit_active_substeps']>0]
        summary={}
        for name,subset in [('all',good),('position_stop_active',active)]:
            summary[name]=dict(count=len(subset),errors={})
            for key in ['root_position_error_m','root_rotation_error_rad','joint_position_error_rad','joint_velocity_error_rad_s','calf_velocity_error_rad_s']:
                summary[name]['errors'][key]={label:dict(maximum=max((r[label][key] for r in subset),default=None),
                    median=float(np.median([r[label][key] for r in subset])) if subset else None) for label in ['old','stops']}
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Source changed during diagnostic '+path)
        result=dict(scope=__doc__,rows=rows,summary=summary,inactive_exact=sum(r['inactive_exact'] is True for r in good),
            intervals=len(rows),completed=len(good),wall_seconds=time.monotonic()-began,source_sha256=pins,
            strict_untouched_isolation_passed=audit['strict_untouched_isolation_passed'],
            new_native_execution=False,valid_expert_labels=False,physical_constraints_changed=False)
        write('result.json',result);write('status.json',dict(state='COMPLETED',intervals=len(rows),completed=len(good),active=len(active)))
        print(json.dumps(plain(summary)),flush=True)
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e)));raise


if __name__=='__main__':main()
