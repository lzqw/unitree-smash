"""Closed-loop prediction from a recorded native state; never native teleportation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.pure_control_separated_recovery_v3 import PureControlSeparatedRecoveryV3 as PureControlRecoveryV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_position_limited_spheres_v1 import predict_position_limited_spheres
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from scripts.event_recovery_manager_v1 import EventRecoveryManagerV1
from scripts.feedback_tail_rollout_v1 import terminal_measures
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['inputs','configuration','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Preserve previous evidence')
    x=read(a.inputs);source=read(read(x['configuration']['model_plan'])['source'])
    *_,audit,ctx=source_context(source['configuration']['source']);initialize(*ctx)
    cfg=read(a.configuration);nominal=read(ROOT/'reports/native_joint_timed71_all128_20261108_02/native_command_inputs.json')['default_joint_position_rad']
    c=PureControlRecoveryV1(PositionPDInterface(**cfg['interface']),nominal);c.spheres=_WORKER['spheres'];state=state_from_row(x['query'])
    geometry=FootSeparationPathConstraintV1(HipMountPathConstraintV1(c.geometry,audit,margin_rad=0.),_WORKER['spheres'],margin_m=0.)
    manager=EventRecoveryManagerV1();rows=[];failure=None;start=time.monotonic()
    pins={str(Path(__file__).resolve()):sha(__file__),str(a.inputs.resolve()):sha(a.inputs),str(a.configuration.resolve()):sha(a.configuration)}
    for name,m in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(m,'__file__',None):pins[str(Path(m.__file__).resolve())]=sha(m.__file__)
    for k in range(80):
        if time.monotonic()-start>120:failure=dict(step=k,reason='wall_budget');break
        r=state.root_rotation_world
        phase=manager.update(episode=0,time_s=3.+k*.01,had_racket_contact=True,released=True,
            racket_touching=False,separated_now=True,ground_normal_force_n=np.where(state.contacts,5.,0.),
            roll_pitch_rad=[np.arctan2(r[2,1],r[2,2]),np.arcsin(np.clip(-r[2,0],-1,1))],
            root_velocity_world=r@state.v[:3],root_angular_velocity_world=r@state.v[3:6],joint_velocity=state.joint_velocity)
        target,d=c.command(state,phase['phase'],3.+k*.01)
        try:prediction=predict_position_limited_spheres(c.model,c.interface,state,target,_WORKER['spheres'],geometry=geometry,substeps=160)
        except (ValueError,np.linalg.LinAlgError) as e:
            failure=dict(step=k,reason='coupled_prediction_rejected',error=repr(e),collision=getattr(geometry,'last_violation',None));break
        bad={key:v for key,v in prediction['diagnostics'].items() if (key=='nonfoot_ground_violation_m' and v>1e-6)
            or (key in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:failure=dict(step=k,reason='physical_model_path',violations=bad,first_ground=prediction['first_ground_violation']);break
        state=prediction['state'];state.previous_command_target=target.copy()
        rows.append(dict(step=k,q=state.q,v=state.v,contacts=state.contacts,target=target,diagnostic=d,
            measures=terminal_measures(_WORKER['controller'],state)))
    for path,h in pins.items():
        if sha(path)!=h:raise ValueError('Changed source '+path)
    output=dict(scope=__doc__,source_sha256=pins,completed_steps=len(rows),required_steps=80,failure=failure,
        trajectory=rows,wall_seconds=time.monotonic()-start,
        contact_semantics='Predictor active ground contacts represented by 5N only to exercise lifecycle; not force validation',
        native_execution=False,valid_expert_labels=False,formal_acceptance=False)
    a.output.write_text(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(steps=len(rows),failure=failure,wall_seconds=output['wall_seconds'])))


if __name__=='__main__':main()
