"""Inspect a model-rejected joint-limit event against actual state/command data.

Read-only: position stop impulses are not added to the model, and original
constraints are not relaxed. Recorded states are sampled at control ticks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-plan','run','bundle','native-audit','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence required')
    read=lambda path:json.loads(Path(path).read_text())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest=read(a.bundle/'manifest.json');audit=read(a.native_audit)
    native=read(a.run/'sequence_native_result.json');setup=read(a.model_plan)
    if read(a.run/'job.json')['status']!='SUCCEEDED' or not audit['passed_dispatch_integrity'] or not native['query_match']['passed']:
        raise ValueError('Completed matching actual run required')
    source=read(setup['source']);_,_,_,_,context=source_context(source['configuration']['source'])
    if context[0]!=manifest['configuration']:raise ValueError('Interface changed')
    initialize(*context);w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    with np.load(a.run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    ticks=np.array([r['tick'] for r in native['records']],int)
    if len(ticks)!=len(trace['q']) or not np.all(np.diff(ticks)==1):raise ValueError('Consecutive states required')
    if np.max(abs(trace['previous_target_rad'][1:]-trace['final_target_rad'][:-1]))>5e-6:
        raise ValueError('Actual command history differs')
    rejection_tick=280+manifest['model_completed_steps']
    lookup={int(t):i for i,t in enumerate(ticks)};rows=[]
    pins=dict(audit['source_sha256'])
    for path in [a.model_plan,a.bundle/'manifest.json',a.native_audit,Path(__file__).resolve()]:pins[str(path.resolve())]=sha(path)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());digest=sha(path)
            if path in pins and pins[path]!=digest:raise ValueError('Pinned source changed '+path)
            pins[path]=digest
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Input changed '+path)
    joint=model.joint_order.index('RL_calf_joint');qi=model.q_indices[joint];vi=model.v_indices[joint]
    for tick in range(rejection_tick-8,rejection_tick+9):
        if tick not in lookup or tick+1 not in lookup:raise ValueError('Requested event window not recorded')
        i=lookup[tick];j=lookup[tick+1]
        state=state_from_row(dict(q=trace['q'][i],v=trace['v'][i],contacts=trace['contacts'][i],target=trace['previous_target_rad'][i]))
        prediction=predict_speed_limited_spheres(model,interface,state,trace['final_target_rad'][i],w['spheres'],geometry=w['geometry'],substeps=160)
        following=prediction['state']
        rows.append(dict(tick=tick,actual_q_rad=float(trace['q'][i,qi]),actual_v_rad_s=float(trace['v'][i,vi]),
            actual_command_rad=float(trace['final_target_rad'][i,joint]),
            measured_next_q_rad=float(trace['q'][j,qi]),predicted_next_q_rad=float(following.q[qi]),
            measured_next_v_rad_s=float(trace['v'][j,vi]),predicted_next_v_rad_s=float(following.v[vi]),
            predicted_joint_position_violation_rad=prediction['diagnostics']['joint_position_violation_rad'],
            predicted_nonfoot_ground_violation_m=prediction['diagnostics']['nonfoot_ground_violation_m']))
    actual_q=trace['q'][:,model.q_indices]
    violation=np.maximum(interface.limits[:,0]-actual_q,actual_q-interface.limits[:,1])
    for path,digest in pins.items():
        if sha(path)!=digest:raise ValueError('Input changed during audit '+path)
    result=dict(scope=__doc__,case=manifest['case'],joint=model.joint_order[joint],joint_limits_rad=interface.limits[joint],
        model_rejection_tick=rejection_tick,rows=rows,
        maximum_sampled_actual_joint_position_violation_rad=float(max(0,violation.max())),
        maximum_sampled_actual_selected_joint_violation_rad=float(max(0,violation[:,joint].max())),
        strict_untouched_isolation_passed=audit['strict_untouched_isolation_passed'],source_sha256=pins,
        native_execution=False,valid_expert_labels=False,
        limitations=['Measured states are 10 ms samples; between-sample excursions are not excluded.',
            'The predictor lacks joint position stop and nonfoot/self-contact impulses.',
            'One-step errors do not alone identify which omitted mechanism caused them.'])
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(plain({k:result[k] for k in ['case','joint','model_rejection_tick','maximum_sampled_actual_joint_position_violation_rad','maximum_sampled_actual_selected_joint_violation_rad']})))


if __name__=='__main__':main()
