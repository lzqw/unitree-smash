"""Compare model impulse, normal-force and full-force-norm contact labels.

The native observed flag uses the norm of net_forces_w with 1/0.5 N
hysteresis. Applying that rule to predicted ground impulses aligns the
threshold arithmetic only; it does not reproduce all native sensor forces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.check_early_support_dispatch_v1 import state_from_arrays
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts import predict_frictional_spheres_v1 as predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['response','run','model-source','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh evidence required')
    source=json.loads(a.response.read_text());pins=dict(source['source_sha256'])
    for path in [a.response,a.model_source,Path(__file__),ROOT/'quadsmash/tasks/stand/coupled_env.py']:
        pins[str(path.resolve())]=sha(path)
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Changed prediction source: '+path)
    run=a.run.resolve();meta=json.loads((run/'sequence_native_result.json').read_text())
    for name in ['sequence_native_result.json','sequence_native_trace.npz']:
        if pins.get(str(run/name))!=sha(run/name):raise ValueError('Wrong native run')
    _,_,_,audit,context=source_context(a.model_source);initialize(*context);w=_WORKER
    model=w['controller'].qp.model
    with np.load(run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    index={r['control_tick']:i for i,r in enumerate(meta['records'])};rows=[]
    original=predictor.frictional_velocity_step
    for recorded in source['one_step']:
        tick=recorded['control_tick'];i=index[tick];j=index[tick+1]
        state=state_from_arrays(model,trace['q'][i],trace['v'][i],trace['contacts'][i],trace['foot_force_world_n'][i])
        state.previous_command_target=trace['previous_target_rad'][i].copy()
        normal=state.contacts.copy();norm=normal.copy();last=None;micro_disagreements=0
        def observe(*args,**kwargs):
            nonlocal normal,norm,last,micro_disagreements
            result=original(*args,**kwargs);last=result['impulse'].reshape(4,3)/args[8]
            normal=np.where(normal,abs(last[:,2])>.5,abs(last[:,2])>1.)
            size=np.linalg.norm(last,axis=1);norm=np.where(norm,size>.5,size>1.)
            micro_disagreements+=int(np.sum(normal!=norm))
            return result
        predictor.frictional_velocity_step=observe
        try:
            pred=predictor.predict_frictional_spheres(model,w['controller'].qp.interface,state,
                trace['final_target_rad'][i],w['spheres'],substeps=160,
                geometry=HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02))
        finally:predictor.frictional_velocity_step=original
        if not np.array_equal(normal,recorded['predicted_force_hysteresis_contacts']):
            raise ValueError('Earlier normal-only diagnostic did not reproduce')
        actual=trace['contacts'][j]
        rows.append(dict(control_tick=tick,native_contacts=actual.tolist(),
            predicted_impulse_contacts=pred['state'].contacts.tolist(),
            normal_hysteresis=normal.tolist(),norm_hysteresis=norm.tolist(),
            normal_disagreements=int(np.sum(normal!=actual)),norm_disagreements=int(np.sum(norm!=actual)),
            normal_vs_norm_micro_foot_disagreements=micro_disagreements,
            predicted_last_ground_force_world_n=last.tolist(),
            native_last_net_force_world_n=trace['foot_force_world_n'][j].tolist()))
    summary=dict(transitions=len(rows),normal_mismatched_intervals=sum(r['normal_disagreements']>0 for r in rows),
        norm_mismatched_intervals=sum(r['norm_disagreements']>0 for r in rows),
        normal_mismatched_feet=sum(r['normal_disagreements'] for r in rows),
        norm_mismatched_feet=sum(r['norm_disagreements'] for r in rows),
        changed_endpoint_label_ticks=[r['control_tick'] for r in rows if r['normal_hysteresis']!=r['norm_hysteresis']],
        changed_micro_label_ticks=[r['control_tick'] for r in rows if r['normal_vs_norm_micro_foot_disagreements']])
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Input changed during comparison')
    a.output.write_text(json.dumps(dict(scope=__doc__,summary=summary,rows=rows,source_sha256=pins,
        control_parameters_changed=False,native_rerun=False,valid_expert_labels=False),indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
