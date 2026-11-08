"""Paired one-control-step prediction against unchanged native state/commands."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain


def pair(job):
    w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    before=state_from_sample(model,job['before'],job['origin']);after=state_from_sample(model,job['after'],job['origin'])
    target=job['after']['previous_target'];rows={};began=time.monotonic()
    for label,predict in [('previous',predict_frictional_spheres),('speed_limited',predict_speed_limited_spheres)]:
        try:
            result=predict(model,interface,before,target,w['spheres'],geometry=w['geometry'],substeps=160)
            state=result['state'];difference=pin.difference(model.model,after.q,state.q)
            rows[label]=dict(completed=True,diagnostics=result['diagnostics'],
                root_position_error_m=float(np.linalg.norm(state.q[:3]-after.q[:3])),
                root_rotation_error_rad=float(np.linalg.norm(difference[3:6])),
                root_linear_velocity_error_m_s=float(np.linalg.norm(state.v[:3]-after.v[:3])),
                root_angular_velocity_error_rad_s=float(np.linalg.norm(state.v[3:6]-after.v[3:6])),
                joint_position_error_rad=(state.q[model.q_indices]-after.q[model.q_indices]),
                joint_velocity_error_rad_s=(state.v[model.v_indices]-after.v[model.v_indices]),
                predicted_joint_velocity=state.v[model.v_indices])
        except ValueError as error:rows[label]=dict(completed=False,error=repr(error))
    distances,_=w['geometry'].evaluate(before.q)
    return dict(case=job['case'],tick=job['tick'],actual_joint_velocity_before=before.joint_velocity,
        actual_joint_velocity_after=after.joint_velocity,actual_previous_command=before.previous_command_target,
        actual_dispatched_target=target,actual_initial_nonfoot_gap_m=float(distances.min()),
        rows=rows,seconds=time.monotonic()-began)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh evidence required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan=read(ROOT/'reports/landing_recovery_family_search_20261108/plan.json')
    bank=Path(plan['bank_run']);follow=Path(plan['bank_followup']);audit=read(follow/'query_bank_audit.json');meta=read(bank/'recovery_query_bank.json')
    if not audit['passed']:raise ValueError('Audited native bank required')
    for path,h in audit['source_sha256'].items():
        if sha(path)!=h:raise ValueError('Native evidence changed')
    source=read(plan['source']);_,_,_,_,context=source_context(source['configuration']['source'])
    preview=read(ROOT/'reports/landing_success_reference_probes_20261108/measured_selection_preview.json')
    donors=sorted({d['case'] for row in preview['rows'] for d in row['donors']});failures=[r['case'] for r in preview['rows']]
    jobs=[]
    with np.load(bank/'recovery_query_bank.npz',allow_pickle=False) as archive:
        z={k:archive[k] for k in archive.files}
        for case in donors+failures:
            for tick in ([280,285,290,300] if case in donors else [280,290,300,310]):
                index=int(np.flatnonzero(z['control_ticks']==tick)[0])
                if not z['unfinished_first_episode'][index:index+2,case].all():raise ValueError('Cross-reset transition')
                jobs.append(dict(case=case,tick=tick,origin=meta['origins_world'][case],
                    before={k:z[k][index,case].copy() for k in z if k!='control_ticks'},
                    after={k:z[k][index+1,case].copy() for k in z if k!='control_ticks'}))
    pins=dict(plan['source_sha256']);pins.update(audit['source_sha256'])
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=str(Path(module.__file__).resolve());pins[path]=sha(path)
    pins[str(Path(__file__).resolve())]=sha(__file__)
    began=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=context) as pool:
        for future in as_completed([pool.submit(pair,job) for job in jobs]):rows.append(future.result())
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Input changed during calibration')
    summary={}
    for label in ['previous','speed_limited']:
        success=[r['rows'][label] for r in rows if r['rows'][label]['completed']]
        errors=np.array([max(abs(s['joint_velocity_error_rad_s'])) for s in success])
        summary[label]=dict(completed=len(success),failed=len(rows)-len(success),
            maximum_joint_velocity_error_rad_s=float(errors.max()) if len(errors) else None,
            median_maximum_joint_velocity_error_rad_s=float(np.median(errors)) if len(errors) else None,
            speed_limit_activated_intervals=sum(s['diagnostics'].get('velocity_limit_active_substeps',0)>0 for s in success))
    result=dict(scope=__doc__,source_sha256=pins,donors=donors,failures=failures,summary=summary,
        rows=sorted(rows,key=lambda r:(r['case'],r['tick'])),seconds=time.monotonic()-began,
        new_native_execution=False,valid_expert_labels=False,
        limits=['Only 10 ms predictions; no full-recovery or real-time claim.',
                'Stored previous target at the next state is the actual intervening command.',
                'Unknown nonfoot/self-collision impulses can still invalidate this foot-only model.'])
    a.output.write_text(json.dumps(plain(result),indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
