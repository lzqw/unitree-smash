"""Diagnose actual/model momentum and foot impulse on clean contact intervals.

Outcome selection is retrospective analysis only, never a control gate. The
native contact view supplies normal forces only; horizontal friction impulse
is inferred from momentum and must not be labeled directly measured friction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.receding_full_command_search_v1 import initialize_court
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.foot_pair_clearance_kinematics_v1 import foot_pair_clearance
import scripts.predict_position_limited_spheres_v1 as prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','model-inputs','one-step-audit','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text())
    if a.output.exists():raise ValueError('Fresh evidence required')
    x=read(a.model_inputs);s=read(read(x['configuration']['model_plan'])['source'])
    *_,context=source_context(s['configuration']['source']);initialize_court(*context)
    w=_WORKER;c=w['controller'];model=c.qp.model;mass=sum(i.mass for i in model.model.inertias)
    def arrays(path):
        with np.load(path) as z:return {k:z[k].copy() for k in z.files}
    data=arrays(a.run/'pure_control_commands.npz');forces=arrays(a.run/'pure_control_ground_forces.npz')
    names=read(a.run/'pure_control_ground_forces.json')['body_names'];feet=[names.index(n) for n in model.feet]
    non=[i for i,n in enumerate(names) if not n.endswith('_foot')];episodes=read(a.run/'episodes.json')
    records=[]
    for row in read(a.one_step_audit)['records']:
        if not row['predicted']:continue
        lane=row['case'];tick=row['control_tick']
        if episodes[lane]['robot_net_fault'] or forces['interval_peak_normal_n'][tick,lane,non].max()>0:continue
        ids=np.flatnonzero((data['lane']==lane)&(data['tick']==tick));nextids=np.flatnonzero((data['lane']==lane)&(data['tick']==tick+1))
        if len(ids)!=1 or len(nextids)!=1:raise ValueError('Actual consecutive state pair required')
        i,j=int(ids[0]),int(nextids[0]);state=state_from_row(dict(q=data['q'][i],v=data['v'][i],target=data['previous'][i],contacts=row['native_contacts']))
        h0=model.snapshot(state.q,state.v)['momentum_world'][:3]
        h1=model.snapshot(data['q'][j],data['v'][j])['momentum_world'][:3]
        native_impulse=forces['interval_normal_impulse_ns'][tick,lane,feet]
        steps=[];original=prediction.position_limited_velocity_step
        def observed(*args,**kwargs):
            result=original(*args,**kwargs)
            steps.append(dict(impulse=result['impulse'].copy(),torque=result['torque'].copy()))
            return result
        prediction.position_limited_velocity_step=observed
        try:pred=prediction.predict_position_limited_spheres(model,c.qp.interface,state,data['final'][i],w['spheres'],geometry=w['geometry'],substeps=160)
        finally:prediction.position_limited_velocity_step=original
        y=pred['state'];hp=model.snapshot(y.q,y.v)['momentum_world'][:3]
        model_impulse=np.sum([r['impulse'].reshape(4,3) for r in steps],axis=0)
        gravity=np.array([0.,0.,-mass*9.81*.01])
        pair_gap,_=foot_pair_clearance(model,state.q,w['spheres'])
        next_pair_gap,_=foot_pair_clearance(model,data['q'][j],w['spheres'])
        records.append(dict(case=lane,tick=tick,model_mass_kg=mass,
            native_ground_normal_impulse_ns=native_impulse,model_ground_impulse_ns=model_impulse,
            native_linear_momentum_change_ns=h1-h0,model_linear_momentum_change_ns=hp-h0,
            native_vertical_balance_residual_ns=float((h1-h0-gravity-native_impulse.sum(0))[2]),
            model_linear_balance_residual_ns=hp-h0-gravity-model_impulse.sum(0),
            native_horizontal_external_impulse_inferred_ns=(h1-h0-gravity)[:2],
            first_foot_pair_min_gap_m=float(pair_gap.min()),next_foot_pair_min_gap_m=float(next_pair_gap.min()),
            model_velocity_error=row['angular_velocity_error_rad_s'],
            native_no_nonfoot_ground_in_interval=True,native_no_robot_net_fault_entire_episode=True,
            unmeasured_self_collision_not_excluded=True))
    out=dict(records=records,normal_force_only=True,native_friction_directly_measured=False,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [a.run/'pure_control_commands.npz',a.run/'pure_control_ground_forces.npz',a.one_step_audit,a.model_inputs,Path(__file__)]})
    a.output.write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n')
    print(json.dumps(plain(records),indent=2))


if __name__=='__main__':main()
