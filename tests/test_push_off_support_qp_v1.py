import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.push_off_support_qp_v1 import PushOffSupportQPV1


def interface():
    return PositionPDInterface(**json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface'])


def test_full_command_interface_retains_original_bounds_without_slew():
    source=interface();qp=PushOffSupportQPV1(source);i=qp.interface
    q=source.limits.mean(1);i.bind_previous_command(q)
    low,high=i.admitted_target_bounds(q,.01)
    np.testing.assert_array_equal(low,source.limits[:,0])
    np.testing.assert_array_equal(high,source.limits[:,1])
    np.testing.assert_array_equal(i.velocity,source.velocity)
    assert qp.preserve_arm_command


def test_flight_recovery_and_other_phases_never_invoke_qp():
    qp=PushOffSupportQPV1(interface());ref=qp.interface.limits.mean(1)
    for phase,recovery,contacts in [('FLIGHT',False,[0]*4),('PUSH_OFF',True,[1]*4),('PREPARE',False,[1]*4),('PUSH_OFF',False,[0]*4)]:
        state=SimpleNamespace(contacts=np.array(contacts,dtype=bool))
        result=qp.propose(state,ref,None,None,None,None,phase=phase,recovery_owned=recovery)
        assert not result['accepted'] and result['owner']=='existing_controller'
        np.testing.assert_array_equal(result['force_world'],np.zeros((4,3)))
        np.testing.assert_array_equal(result['joint_position_target'],ref)


def test_push_task_keeps_causal_angular_baseline_without_upright_servo():
    qp=PushOffSupportQPV1(interface());baseline=np.arange(24)*.1
    com=np.array([1.,2.,3.]);delta=np.array([.2,0.,.5]);angular=np.array([0.,.4,0.])
    qp._task=(baseline,com,delta,angular)
    result=qp.reference(None,{'com_world':np.array([0.,0.,.3])},None)
    np.testing.assert_array_equal(result['com_acceleration'],com+delta)
    np.testing.assert_array_equal(result['angular_acceleration_body'],baseline[3:6]+angular)
    np.testing.assert_array_equal(result['joint_acceleration'],baseline[qp.model.v_indices])


def test_push_proposal_uses_actual_implicit_command_map_and_keeps_arm():
    from test_command_history_support_v1 import fixture
    _,source,state=fixture();state.previous_command_target=state.joint_position.copy()
    qp=PushOffSupportQPV1(source)
    plan=qp.propose(state,state.joint_position,np.zeros(24),np.zeros(3),np.array([0.,0.,.2]),np.zeros(3),phase='PUSH_OFF')
    assert plan['accepted'],plan
    np.testing.assert_allclose(plan['joint_position_target'][12:],state.joint_position[12:],atol=2e-6)
    assert plan['residuals']['implicit_pd_torque']<2e-4
    assert plan['residuals']['dynamics']<2e-4
    assert not plan['fullbody_prediction_valid'] and not plan['valid_teacher_reference']
    assert qp._task is None
