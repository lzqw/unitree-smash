from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
from scripts.hip_mount_planning_constraints_v1 import HipMountPlanningConstraintsV1


def fixture():
    record=json.loads((Path(__file__).resolve().parents[1]/'configs/go2_piper_racket_landing_model_v1.json').read_text())
    return HipMountPlanningConstraintsV1(SimpleNamespace(joint_order=record['joint_order']))


def test_collision_rows_constrain_predicted_state_not_target_command():
    geometry=fixture()
    # PD needs a position error to exert effort. A target past the collision
    # boundary can still be a valid command if the predicted state stays clear.
    stage=dict(joint_position_map=np.eye(18)*.1,joint_position_bias=np.zeros(18))
    A,low,high=geometry.interval([stage]);command=np.zeros(18);command[0]=1.
    assert np.all(A@command>=low) and np.all(A@command<=high)
    stage['joint_position_map']=np.eye(18)
    A,low,high=geometry.interval([stage])
    assert np.any(A@command<low)


def test_inward_and_outward_motion_receive_different_braking_targets():
    geometry=fixture();q=np.zeros(18);q[0]=.55
    outward=np.zeros(18);outward[0]=2.;inward=-outward
    m,b,dout=geometry.stopping_objective(q,outward,np.eye(18),np.zeros(18))
    mi,bi,din=geometry.stopping_objective(q,inward,np.eye(18),np.zeros(18))
    assert dout[0]>din[0]
    assert m[0,0]==-1 # positive gap acceleration means braking the left hip
