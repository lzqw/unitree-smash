import json
from pathlib import Path
import numpy as np
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.compare_recorded_predictors_v2 import measured_state
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.experimental_sphere_admission_v1 import ExperimentalSphereAdmissionV1
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres


def test_wrench_audit_does_not_change_dynamics_or_inputs():
    model=LoadedPinocchioModelV1()
    interface=PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])
    with np.load('reports/task_constrained_hit_20261118/attempts/calibration01/native/control_state.npz') as z:
        state=measured_state(model,z,249,1);target=z['previous_target'][250,1].copy()
    q=state.q.copy();v=state.v.copy();previous=state.previous_command_target.copy()
    geometry=AuthoredGroundClearance('configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    adapter=ExperimentalSphereAdmissionV1(model,geometry,'reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    kwargs=dict(substeps=160,geometry=geometry,enforce_joint_stops=True,
        diagnostic_nonfoot_spheres=adapter.extras,diagnostic_hip_mount_response=adapter.hip,contact_max_iterations=2000)
    plain=predict_speed_limited_spheres(model,interface,state,target,adapter.spheres,**kwargs)
    audited=predict_speed_limited_spheres(model,interface,state,target,adapter.spheres,audit_wrench=True,**kwargs)
    np.testing.assert_array_equal(plain['state'].q,audited['state'].q)
    np.testing.assert_array_equal(plain['state'].v,audited['state'].v)
    assert plain['diagnostics']==audited['diagnostics']
    assert plain['wrench_audit'] is None
    w=audited['wrench_audit']
    assert w['contact_impulse_world_ns'].shape==(len(adapter.spheres)+len(adapter.extras),3)
    assert np.isfinite(w['contact_angular_impulse_com_world_nms']).all()
    assert np.all((w['saturated_substeps']>=0)&(w['saturated_substeps']<=160))
    assert np.all(abs(w['drive_torque_integral_nms'])<=interface.effort*.01+1e-8)
    np.testing.assert_array_equal(state.q,q);np.testing.assert_array_equal(state.v,v)
    np.testing.assert_array_equal(state.previous_command_target,previous)
    # Integrated external contact moment matches the fullbody COM momentum
    # change up to finite-step integration error, independently of solver residual.
    before=model.snapshot(q,v);after=model.snapshot(audited['state'].q,audited['state'].v)
    dh=after['momentum_world'][3:]-before['momentum_world'][3:]
    assert np.linalg.norm(dh-w['contact_angular_impulse_com_world_nms'].sum(0))<.002
