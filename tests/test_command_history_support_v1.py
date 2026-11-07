"""Compare command admission with production slew; retain physical bounds."""
import numpy as np
import pytest
import torch
from scripts.command_history_support_qp_v1 import CommandHistoryPDInterface
from scripts.landing_command_history_mpc_v1 import LandingCommandHistoryMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from quadsmash.control.full_racket_arm_target_rate_v4 import ArmTargetRateV4
from test_admitted_arm_support_qp_v1 import fixture


@pytest.mark.parametrize("rate",[5.,10.])
def test_history_interface_matches_original_arm_slew_with_large_tracking_lag(rate):
    _, source, state=fixture()
    interface=CommandHistoryPDInterface(source,.01,rate)
    original=ArmTargetRateV4(1,'cpu',rate)
    previous=state.joint_position.copy();previous[12:]+=.5
    original.target[0]=torch.tensor(previous[12:]);original.initialized[:]=True;original.clock[:]=.3
    actual=torch.tensor(state.joint_position[12:][None],dtype=torch.float32)
    for tick in range(1,4):
        desired=previous.copy();desired[12:]+=[.2,-.2,.1,-.1,.3,-.3]
        interface.bind_previous_command(previous)
        low,high=interface.admitted_target_bounds(state.joint_position,.01)
        admitted=np.clip(desired,low,high)
        native=original(torch.tensor(desired[12:][None],dtype=torch.float32),actual,
                        torch.tensor([.3+.01*tick]))[0].numpy()
        np.testing.assert_allclose(admitted[12:],native,atol=5e-7)
        assert np.max(abs(admitted[12:]-state.joint_position[12:]))>.4
        previous=admitted


def test_target_error_is_not_artificial_torque_limit_and_saturation_is_realized():
    _, source, state=fixture();interface=CommandHistoryPDInterface(source,.01,5.)
    q=np.zeros(18);v=np.zeros(18);previous=np.zeros(18);previous[12:]=2.
    interface.bind_previous_command(previous)
    low,high=interface.torque_bounds(q,v,.01)
    np.testing.assert_allclose(low[:12],-23.5);np.testing.assert_allclose(high[:12],23.5)
    # Both arm interval endpoints saturate: this remains a realizable singleton.
    np.testing.assert_allclose(low[12:],100.);np.testing.assert_allclose(high[12:],100.)
    torque=np.r_[np.full(12,20.),np.full(6,100.)]
    target,realized=interface.target_from_torque(torque,q,v)
    np.testing.assert_allclose(realized,torque,atol=1e-10)
    assert np.all(target[:12]>.3)
    assert np.max(abs(realized))<=100.


def test_missing_history_fails_instead_of_substituting_measured_position():
    _, source, state=fixture();interface=CommandHistoryPDInterface(source,.01,5.)
    with pytest.raises(ValueError,match='Command history'):
        interface.admitted_target_bounds(state.joint_position,.01)


def test_qp_and_rollout_carry_final_commands_without_relaxing_velocity_limits():
    _,source,state=fixture();state.previous_command_target=state.joint_position.copy()
    mpc=LandingCommandHistoryMPCV1(source,settings=LandingMPCSettingsV2(
        horizon_steps=1,executor_replay=True),preserve_arm_command=False)
    mpc.observe(state,state.joint_position,lane=1,control_tick=0)
    plan=mpc.qp.solve(state,state.joint_position)
    assert plan['accepted'],plan.get('rejection')
    assert np.all(abs(plan['predicted_joint_velocity'])<=source.velocity+3e-4)
    following=mpc._integrate(state,plan['acceleration'],plan)
    np.testing.assert_allclose(following.previous_command_target,plan['joint_position_target'],atol=1e-10)
    np.testing.assert_array_equal(state.previous_command_target,state.joint_position)
    terminal,_=mpc.replay_horizon(state,[{k:v.tolist() for k,v in plan.items()
        if k in ('acceleration','torque','joint_position_target')}])
    np.testing.assert_allclose(terminal.q,following.q,atol=1e-12)
    np.testing.assert_allclose(terminal.previous_command_target,following.previous_command_target,atol=1e-12)
    mpc.qp.reset();assert mpc.qp.interface.previous_command is None


def test_native_evaluator_factory_uses_checkpoint_slew_without_isaac_startup():
    # Execute the actual factory method in isolation; do not start a simulator
    # just to verify configuration plumbing and constructor availability.
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from dataclasses import replace
    from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings
    from scripts.landing_support_mpc_v1 import LandingSupportMPCV1,LandingMPCSettingsV1
    from scripts.landing_support_mpc_v2 import LandingSupportMPCV2
    from scripts.landing_support_mpc_v3 import LandingSupportMPCV3
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py'
    tree=ast.parse(path.read_text());values={}
    for node in tree.body:
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):continue
        call=node.value
        if not isinstance(call.func,ast.Attribute) or call.func.attr!='add_argument':continue
        if not call.args or not isinstance(call.args[0],ast.Constant):continue
        name=call.args[0].value
        if not isinstance(name,str) or not name.startswith('--landing-mpc-'):continue
        kw={x.arg:x.value for x in call.keywords}
        if 'default' in kw:values[name[2:].replace('-','_')]=ast.literal_eval(kw['default'])
        elif 'action' in kw and ast.literal_eval(kw['action'])=='store_true':values[name[2:].replace('-','_')]=False
    values.update(landing_mpc_cases={1},landing_mpc_command_history_interface=True,
        landing_mpc_continuous_reference=True,landing_mpc_support_margin=True,
        landing_mpc_executor_replay=True,landing_mpc_arm_target_rate=10.,_landing_mpc_horizon_steps=2)
    subject=SimpleNamespace(**values)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SharedTargetResidualController')
    factory=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='configure_landing_mpc')
    namespace=dict(np=np,torch=torch,replace=replace,PositionPDInterface=PositionPDInterface,
        SupportSettings=SupportSettings,LandingCommandHistoryMPCV1=LandingCommandHistoryMPCV1,
        LandingSupportMPCV3=LandingSupportMPCV3,LandingSupportMPCV2=LandingSupportMPCV2,
        LandingMPCSettingsV2=LandingMPCSettingsV2,LandingSupportMPCV1=LandingSupportMPCV1,
        LandingMPCSettingsV1=LandingMPCSettingsV1)
    exec(compile(ast.Module(body=[factory],type_ignores=[]),str(path),'exec'),namespace)
    namespace['configure_landing_mpc'](subject,np.tile([-4.,4.],(18,1)),np.zeros(18))
    assert isinstance(subject.landing_mpc,LandingCommandHistoryMPCV1)
    assert subject.landing_mpc.qp.interface.arm_rate==10.
    np.testing.assert_array_equal(subject.landing_mpc.qp.interface.velocity[12:],np.full(6,5.))
