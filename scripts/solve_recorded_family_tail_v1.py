"""Optimize sustained recovery from a later actually reached student state.

The student prefix is real ground-start data, not a simulated old teacher
prefix or a teleported PhysX state. An accepted model plan still needs native
full-ground execution. Historical net-contact flags are explicitly identified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.recovery_family_queries_v1 import transfer_reference
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_recovery_planner_v3 import CurrentStateRecoveryPlannerV3
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','search','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--case',type=int,required=True);p.add_argument('--tick',type=int,required=True)
    a=p.parse_args()
    if a.output.exists() or not 350<=a.tick<=550:raise ValueError('Fresh bounded recovery query required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    setup,search=read(a.plan),read(a.search)
    selected=[r for r in read(setup['selection'])['selected'] if r['case']==a.case]
    if len(selected)!=1 or not search['verified'] or selected[0]['release_time_s']>=a.tick*.01:
        raise ValueError('Preselected released-ball family and complete warm model required')
    bank=Path(setup['bank_run']);meta=read(bank/'recovery_query_bank.json')
    admission=read(Path(setup['bank_followup'])/'query_bank_audit.json')
    if not admission['passed'] or not meta['complete_window'] or sha(bank/'recovery_query_bank.npz')!=meta['data_sha256']:
        raise ValueError('Complete immutable passive native bank required')
    source=read(setup['source']);*_,geometry,context=source_context(source['configuration']['source'])
    initialize(*context);w=_WORKER;model=w['controller'].qp.model
    with np.load(bank/'recovery_query_bank.npz',allow_pickle=False) as z:
        def sample(tick):
            indices=np.flatnonzero(z['control_ticks']==tick)
            if len(indices)!=1:raise ValueError('Missing unique native control clock')
            row={k:z[k][indices[0],a.case].copy() for k in z.files if k!='control_ticks'}
            if row['episode_index']!=0 or not row['unfinished_first_episode']:
                raise ValueError('Actual first episode already finished')
            return state_from_sample(model,row,meta['origins_world'][a.case])
        initial=sample(280);state=sample(a.tick)
    constraint=HipMountPathConstraintV1(w['geometry'],geometry,margin_rad=.02)
    gaps,_=constraint.evaluate(state.q)
    if gaps.min() < -1e-6:raise ValueError('Initial native state already violates modeled nonfoot clearance')
    context,_=transfer_reference(context,state_from_row(search['query']))
    context,alignment=transfer_reference(context,initial)
    warm=dict(contract='foot_space_current_state_plan_v1',original_knots_rad=search['original_knots'],
        transition_knots_rad=search['verified']['replay']['transition_knots'],braking=search['braking'],corrections=[])
    pins=dict(search['source_sha256']);pins.update(admission['source_sha256'])
    for f in [a.plan,a.search,Path(setup['selection']),bank/'recovery_query_bank.json',bank/'recovery_query_bank.npz',Path(__file__)]:
        pins[str(f.resolve())]=sha(f)
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed source '+f)
    planner=CurrentStateRecoveryPlannerV3(context,geometry,workers=2,max_seconds=600,seed=882122)
    try:result=planner.solve(state,a.tick,warm,preceding_stable_ticks=0)
    finally:planner.close()
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Pinned source changed during optimization '+f)
    result.update(case=a.case,selection=selected[0],source_sha256=pins,reference_alignment=alignment,
        initial_minimum_nonfoot_gap_m=float(gaps.min()),actual_student_prefix=True,
        query_contact_source='original net-force hysteresis; not the new independently filtered ground sensor',
        native_execution=False,valid_expert_labels=False,simulator_state_reset=False,physical_impossibility_claim=False)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print(json.dumps(dict(case=a.case,tick=a.tick,accepted=result['accepted'],wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
