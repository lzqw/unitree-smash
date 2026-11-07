import numpy as np
from scripts.condensed_implicit_drive_v1 import condense_implicit_drive,interval_constraints
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from test_admitted_arm_support_qp_v1 import fixture


def test_loaded_robot_condensation_matches_independent_substep_integration():
    qp,interface,state=fixture();model=qp.model
    snap=model.snapshot(state.q,state.v);feet=np.flatnonzero(state.contacts)
    jac=snap['foot_jacobian'][feet].reshape(-1,24)
    jdot=(snap['foot_jacobian_dot'][feet]@state.v).reshape(-1)
    selection=np.zeros((24,18));selection[model.v_indices,np.arange(18)]=1.
    target=state.joint_position.copy();target[12:]+=.001
    stages=condense_implicit_drive(snap['mass_matrix'],snap['nonlinear'],selection,
        interface.kp,interface.kd,jac,jdot,state.v,state.joint_position,substeps=16)
    final=stages[-1]
    predicted=predict_implicit_support(model,interface,state,target,substeps=16)
    np.testing.assert_allclose(final['velocity_map']@target+final['velocity_bias'],predicted['state'].v,atol=1e-9)
    np.testing.assert_allclose(final['joint_position_map']@target+final['joint_position_bias'],
        predicted['state'].joint_position,atol=1e-10)


def test_interval_limits_detect_fast_transient_hidden_by_endpoint():
    selection=np.ones((1,1))
    stages=condense_implicit_drive(np.array([[.001]]),np.zeros(1),selection,
        np.ones(1),np.full(1,10.),np.zeros((0,1)),np.zeros(0),np.array([2.]),np.zeros(1),substeps=16)
    target=np.zeros(1)
    first=stages[0]['velocity_map']@target+stages[0]['velocity_bias']
    last=stages[-1]['velocity_map']@target+stages[-1]['velocity_bias']
    assert first[0]>.2 and abs(last[0])<.2
    constraints=interval_constraints(stages,selection,effort=np.full(1,100.),
        velocity_limit=np.full(1,.2),joint_limits=np.array([[-1.,1.]]),friction=.65)
    response=constraints['matrix']@target
    violating=(response<constraints['low']-1e-8)|(response>constraints['high']+1e-8)
    assert any(violating)
    assert any(label[1]=='joint_velocity' and label[0]==0 for label,bad in zip(constraints['labels'],violating) if bad)
