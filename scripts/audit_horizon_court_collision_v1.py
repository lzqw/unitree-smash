"""Recheck a completed candidate with original net geometry at every model microstep."""
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
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1,CourtCollisionPathConstraintV1
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from scripts.predict_position_limited_spheres_v1 import predict_position_limited_spheres
from scripts.audit_joint_sequence_proposal_v1 import plain
from quadsmash.physics.court_v4 import load_court


def main():
    p=argparse.ArgumentParser();
    for n in ['inputs','search','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh output required')
    x=read(a.inputs);s=read(read(x['configuration']['model_plan'])['source']);*_,audit,ctx=source_context(s['configuration']['source']);initialize(*ctx)
    c=_WORKER['controller'];state=state_from_row(x['query']);search=read(a.search)
    net=AuthoredNetClearanceV1(_WORKER['geometry'],load_court());geometry=CourtCollisionPathConstraintV1(
        FootSeparationPathConstraintV1(HipMountPathConstraintV1(_WORKER['geometry'],audit,margin_rad=0.),_WORKER['spheres'],margin_m=0.),net)
    expected=search['best']['trajectory'];failure=None;checked=0;max_state_error=0.;started=time.monotonic()
    for k,row in enumerate(expected):
        try:pred=predict_position_limited_spheres(c.qp.model,c.qp.interface,state,np.asarray(row['target']),_WORKER['spheres'],geometry=geometry,substeps=160)
        except ValueError as e:failure=dict(step=k,error=repr(e),collision=geometry.last_violation);break
        state=pred['state'];state.previous_command_target=np.asarray(row['target']).copy()
        max_state_error=max(max_state_error,float(abs(state.q-np.asarray(row['q'])).max()),float(abs(state.v-np.asarray(row['v'])).max()))
        checked+=1
    result=dict(checked_control_steps=checked,required_control_steps=len(expected),microsteps_per_control=160,
        failure=failure,maximum_original_state_replay_error=max_state_error,wall_seconds=time.monotonic()-started,
        court_path_passed=failure is None,native_execution=False,valid_expert_labels=False,
        source_sha256={str(p.resolve()):sha(p) for p in [a.inputs,a.search,Path(__file__),ROOT/'scripts/authored_net_clearance_v1.py',ROOT/'configs/court_net_v4.json']})
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');print(json.dumps(plain(result),allow_nan=False))


if __name__=='__main__':main()
