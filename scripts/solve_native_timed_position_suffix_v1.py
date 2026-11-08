"""Solve a complete timed recovery suffix from one measured original-run state."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.current_state_timed_recovery_planner_v4 import CurrentStateTimedRecoveryPlannerV4
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','run','bundle','native-audit','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tick',type=int,required=True);p.add_argument('--max-seconds',type=float,default=600.)
    a=p.parse_args()
    if a.output.exists() or not 280<=a.tick<=700 or not 120<=a.max_seconds<=900:raise ValueError('Fresh bounded measured-state solve required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    setup=read(a.model_plan);manifest=read(a.bundle/'manifest.json');audit=read(a.native_audit)
    if read(a.run/'job.json')['status']!='SUCCEEDED' or not audit['passed_dispatch_integrity']:raise ValueError('Completed actual run required')
    source=read(setup['source']);_,_,_,geometry,context=source_context(source['configuration']['source'])
    context=list(copy.deepcopy(context));initial=manifest['query']
    if context[0]!=manifest['configuration']:raise ValueError('Original physical configuration differs')
    context[1].update(initial_q=initial['q'],initial_v=initial['v'],initial_contacts=initial['contacts'],actual_previous_command=initial['target'],tick=280)
    with np.load(a.bundle/manifest['bundle_name'],allow_pickle=False) as z:
        context[4]=z['tracking_reference_q'].copy();context[5]=z['tracking_reference_v'].copy()
    context[6]=manifest['tracking_gains']
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    ticks=np.array([r['tick'] for r in read(a.run/'sequence_native_result.json')['records']],int)
    if not np.array_equal(ticks,np.arange(280,800)):raise ValueError('Full actual feedback record required')
    if np.max(abs(trace['previous_target_rad'][1:]-trace['final_target_rad'][:-1]))>5e-6:raise ValueError('Actual command history differs')
    i=a.tick-280;query=dict(q=trace['q'][i],v=trace['v'][i],contacts=trace['contacts'][i],target=trace['previous_target_rad'][i])
    initialize(*context)
    state=state_from_row(query);pins=dict(audit['source_sha256'])
    for path in [a.model_plan,a.bundle/'manifest.json',a.bundle/manifest['bundle_name'],a.native_audit,Path(__file__).resolve()]:pins[str(path.resolve())]=sha(path)
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
    write('inputs.json',dict(configuration={k:str(v) for k,v in vars(a).items()},query=query,source_sha256=pins,
        max_seconds=a.max_seconds,workers=2,new_native_execution=False,physical_constraints_changed=False))
    planner=CurrentStateTimedRecoveryPlannerV4(tuple(context),geometry,workers=2,max_seconds=a.max_seconds,seed=882180)
    began=time.monotonic()
    try:
        write('status.json',dict(state='OPTIMIZING_MEASURED_REMAINDER',tick=a.tick))
        result=planner.solve(state,a.tick,manifest['proposal'])
        for path,digest in pins.items():
            if sha(path)!=digest:raise ValueError('Input changed during solve '+path)
        write('result.json',dict(scope=__doc__,decision=result,source_sha256=pins,wall_seconds=time.monotonic()-began,
            native_outcome=audit['selected_case_outcome'],strict_untouched_isolation_passed=audit['strict_untouched_isolation_passed'],
            native_execution=False,valid_expert_labels=False,physical_constraints_changed=False))
        write('status.json',dict(state='COMPLETED',accepted=result['accepted'],tick=a.tick))
        print(json.dumps(dict(accepted=result['accepted'],tick=a.tick,seconds=time.monotonic()-began)),flush=True)
    except BaseException as e:write('status.json',dict(state='FAILED',error=repr(e)));raise
    finally:planner.close()


if __name__=='__main__':main()
