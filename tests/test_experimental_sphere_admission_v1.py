import json
from pathlib import Path
import numpy as np
import pytest
from scripts.experimental_sphere_admission_v1 import admission_diagnostics
from scripts.compare_recorded_predictors_v2 import measured_state
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.contact_first_fullbody_evaluator_v1 import ContactFirstFullBodyEvaluatorV1


def evidence():
    return dict(ground_geometry_checked=True, diagnostics=dict(momentum_residual=1e-9,
        normal_impulse_violation_ns=1e-9, friction_cone=1e-9,
        normal_complementarity=0., tangent_fixed_point=0., speed_constraint_residual=0.,
        joint_velocity_violation_rad_s=0., joint_position_violation_rad=0.,
        motor_torque_violation_nm=0., nonfoot_ground_violation_m=0., nonfoot_contact_substeps=0))


def test_impulse_residual_is_not_silently_treated_as_force():
    raw=evidence(); d=admission_diagnostics(raw,.0000625)
    assert d['dynamics_residual_max']==pytest.approx(1.6e-5)
    assert d['normal_force_violation_n']==pytest.approx(1.6e-5)
    assert d['friction_violation_n']==pytest.approx(1.6e-5)
    assert 'dynamics_residual_max' not in raw['diagnostics']
    raw['diagnostics']['momentum_residual']=2e-9
    assert 'dynamics_residual_max' in ContactFirstFullBodyEvaluatorV1._violations(admission_diagnostics(raw,.0000625))


@pytest.mark.parametrize('key',list(evidence()['diagnostics']))
@pytest.mark.parametrize('bad',[None,float('nan'),-1.])
def test_missing_or_nonfinite_diagnostics_never_pass(key,bad):
    result=evidence()
    if bad is None:del result['diagnostics'][key]
    else:result['diagnostics'][key]=bad
    with pytest.raises(ValueError):admission_diagnostics(result,.0000625)


def test_unchecked_geometry_and_unconverged_law_rejected():
    raw=evidence();raw['ground_geometry_checked']=False
    with pytest.raises(ValueError):admission_diagnostics(raw,.0000625)
    raw=evidence();raw['diagnostics']['normal_complementarity']=3e-7
    with pytest.raises(ValueError):admission_diagnostics(raw,.0000625)


def test_real_history_through_explicit_adapter_preserves_state():
    model=LoadedPinocchioModelV1()
    cfg=json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface']
    with np.load('reports/task_constrained_hit_20261118/attempts/calibration01/native/control_state.npz') as z:
        state=measured_state(model,z,249,1);target=z['previous_target'][250,1].copy()
    before_q=state.q.copy();before_previous=state.previous_command_target.copy()
    ev=ContactFirstFullBodyEvaluatorV1(interface_config=cfg,initial_state=state,model=model,
        contact_prediction='experimental_spheres',enforce_joint_stops=True,substeps=160,
        experimental_sphere_manifest='reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    result=ev.predict(model,ev.interface,state,target,duration_s=.01,substeps=160,enforce_joint_stops=True)
    assert result['ground_geometry_checked'] and not result['valid_teacher']
    assert not result['native_execution']
    assert not ev._violations(result['diagnostics'])
    np.testing.assert_array_equal(state.q,before_q)
    np.testing.assert_array_equal(state.previous_command_target,before_previous)
    np.testing.assert_array_equal(result['state'].previous_command_target,target)


@pytest.mark.parametrize('diagnostic,value',[
    ('nonfoot_ground_violation_m',2e-6),('nonfoot_contact_substeps',1)])
def test_interior_collision_cannot_pass_with_clean_terminal_geometry(diagnostic,value):
    from copy import deepcopy
    model=LoadedPinocchioModelV1()
    cfg=json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface']
    with np.load('reports/task_constrained_hit_20261118/attempts/calibration01/native/control_state.npz') as z:
        state=measured_state(model,z,249,1)
    # Force a clean boundary but retain an interior-microstep collision report.
    state.q[2]+=2.;state.root_position_world[2]+=2.
    ev=ContactFirstFullBodyEvaluatorV1(interface_config=cfg,initial_state=state,model=model,
        contact_prediction='experimental_spheres',enforce_joint_stops=True,
        experimental_sphere_manifest='reports/prediction_model_validation_20261118/AUTHORED_SPHERES.json')
    raw=evidence();raw['diagnostics'][diagnostic]=value
    ev.predict=lambda *a,**kw: dict(state=deepcopy(state),diagnostics=admission_diagnostics(raw,.0025))
    arm=np.repeat(state.joint_position[None,12:],2,axis=0)
    result=ev(arm,np.zeros_like(arm),1,{})
    assert not result['accepted']
    assert result['rejection']=='full_body_prefix_violation'
    assert result['violations'][diagnostic]==value
    assert result['geometry_witness_after']['gap_m']>0
