"""Attribute reproduced one-step errors to joints and observed contact changes.

Uses completed native execution and its original sampled transitions. No
parameters are fitted here; association does not establish a failure cause.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_actual_recovery_response_v1 import state_error
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['response','run','model-source','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists(): raise ValueError('Preserve previous diagnostic')
    source=json.loads(a.response.read_text());pins=dict(source['source_sha256'])
    for path in [a.response,a.model_source,Path(__file__)]:pins[str(path.resolve())]=sha(path)
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Changed source: '+path)
    if source['summary']['computed_transitions']!=107 or source['summary']['rejected_transitions']:
        raise ValueError('Require completed original 107-transition audit')
    run=a.run.resolve();meta=json.loads((run/'sequence_native_result.json').read_text())
    for name in ['sequence_native_result.json','sequence_native_trace.npz','job.json']:
        path=run/name
        if pins.get(str(path))!=sha(path):raise ValueError('Run differs from response audit: '+name)
    _,_,_,audit,context=source_context(a.model_source);initialize(*context);w=_WORKER
    model=w['controller'].qp.model;interface=w['controller'].qp.interface
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    index={r['control_tick']:i for i,r in enumerate(meta['records'])}
    began=time.monotonic();rows=[];max_repeat_error=0.
    for recorded in source['one_step']:
        if time.monotonic()-began>300:raise TimeoutError('Declared CPU diagnostic limit')
        tick=recorded['control_tick'];i=index[tick];j=index[tick+1]
        state=state_from_arrays(model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
        state.previous_command_target=trace['previous_target_rad'][i].copy()
        geometry=HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02)
        pred=predict_frictional_spheres(model,interface,state,trace['final_target_rad'][i],w['spheres'],
            substeps=160,geometry=geometry)['state']
        errors=state_error(model,pred.q,pred.v,trace['q'][j],trace['v'][j])
        difference=max(abs(errors[k]-recorded['errors'][k]) for k in errors)
        if difference>1e-12:raise ValueError('Original one-step response did not reproduce')
        max_repeat_error=max(max_repeat_error,difference)
        dv=pred.v[model.v_indices]-trace['v'][j,model.v_indices]
        dq=pin.difference(model.model,trace['q'][j],pred.q)[model.v_indices]
        joint=int(np.argmax(abs(dv)))
        rows.append(dict(control_tick=tick,largest_velocity_error_joint=model.joint_order[joint],
            per_joint_velocity_error_rad_s=dv.tolist(),per_joint_position_error_rad=dq.tolist(),
            errors=errors,native_contact_changed=bool(np.any(trace['contacts'][i]!=trace['contacts'][j])),
            native_contacts_before=trace['contacts'][i].tolist(),native_contacts_after=trace['contacts'][j].tolist(),
            predicted_contacts=pred.contacts.tolist(),
            aligned_contact_disagreements=recorded['aligned_contact_disagreements'],
            actual_next_joint_velocity_rad_s=trace['v'][j,model.v_indices].tolist(),
            actual_next_foot_force_world_n=trace['foot_force_world_n'][j].tolist()))
    groups={}
    for label,subset in [('all',rows),('contact_changed',[r for r in rows if r['native_contact_changed']]),
        ('contact_unchanged',[r for r in rows if not r['native_contact_changed']]),
        ('contact_label_disagrees',[r for r in rows if r['aligned_contact_disagreements']]),
        ('contact_label_agrees',[r for r in rows if not r['aligned_contact_disagreements']])]:
        groups[label]=dict(count=len(subset),joint_velocity_max_error_median=float(np.median([
            r['errors']['joint_velocity_max_rad_s'] for r in subset])) if subset else None,
            joint_velocity_max_error_peak=max((r['errors']['joint_velocity_max_rad_s'] for r in subset),default=None))
    matrix=np.asarray([r['per_joint_velocity_error_rad_s'] for r in rows])
    per_joint=[dict(joint=name,median_abs_error=float(np.median(abs(matrix[:,i]))),
        maximum_abs_error=float(np.max(abs(matrix[:,i]))),
        largest_error_count=sum(r['largest_velocity_error_joint']==name for r in rows))
        for i,name in enumerate(model.joint_order)]
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Input changed during audit: '+path)
    result=dict(scope=__doc__,source_sha256=pins,rows=rows,groups=groups,per_joint=per_joint,
        maximum_original_error_reproduction_difference=max_repeat_error,
        wall_seconds=time.monotonic()-began,native_rerun=False,parameters_changed=False,
        physical_cause_identified=False,valid_expert_labels=False,
        limitations=['Only the original selected transitions from TRAIN3 are examined.',
            'Native contact labels are force-hysteresis observations, not independent geometry or load capacity.',
            'Unchanged endpoint contact labels do not exclude a contact event inside the interval.',
            'Joint error concentration and contact association are not causal attribution.'])
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(groups=groups,top_joints=sorted(per_joint,key=lambda r:r['maximum_abs_error'],reverse=True)[:5],
        reproduction_difference=max_repeat_error,wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':main()
