import json
from pathlib import Path
import numpy as np
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.compare_recorded_predictors_v2 import measured_state
from scripts.diagnostic_nonfoot_spheres_v1 import nonfoot_spheres
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_prediction_contact_geometry_v2 import racket_contact_state
import pinocchio as pin


def test_contact_direction_does_not_alias_pinocchio_workspace():
    model=LoadedPinocchioModelV1();q=pin.neutral(model.model);v=np.zeros(model.model.nv)
    first=racket_contact_state(model,q,v,[.0256,0,.07]);saved=first[1].copy()
    q[3:7]=pin.Quaternion(pin.exp3(np.array([.3,.4,.5]))).coeffs()
    second=racket_contact_state(model,q,v,[.0256,0,.07])
    np.testing.assert_array_equal(first[1],saved)
    assert np.linalg.norm(first[1]-second[1])>.1


def test_historical_nonfoot_response_keeps_foot_mask_history_and_admission_separate():
    model=LoadedPinocchioModelV1()
    interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    geometry=AuthoredGroundClearance('configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    feet=json.loads(Path('reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json').read_text())['spheres']
    with np.load('reports/task_constrained_hit_20261118/attempts/calibration01/native/control_state.npz') as data:
        state=measured_state(model,data,231,0)
        target=data['previous_target'][232,0].copy();previous=state.previous_command_target.copy()
        result=predict_speed_limited_spheres(model,interface,state,target,feet,substeps=160,
            enforce_joint_stops=True,diagnostic_nonfoot_spheres=nonfoot_spheres(geometry,model.feet))
    assert result['diagnostic_nonfoot_bodies']==['Head_lower']
    assert result['diagnostics']['nonfoot_normal_impulse_ns']>0
    assert result['state'].contacts.shape==(4,)
    assert not result['valid_teacher'] and result['nonfoot_contact_not_admissible']
    np.testing.assert_array_equal(result['state'].previous_command_target,target)
    result['state'].previous_command_target[:]=123
    np.testing.assert_array_equal(state.previous_command_target,previous)
    assert np.all(target!=123)
