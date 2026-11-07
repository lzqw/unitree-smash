import numpy as np
import pytest
from scripts.audit_native_command_cost_v1 import validate_selection
from scripts.landing_feasible_command_cost_mpc_v1 import LandingFeasibleCommandCostMPCV1
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import SupportSettings
from test_admitted_arm_support_qp_v1 import fixture


def test_failed_candidate_does_not_advance_reference_twice():
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    # Deliberately strict terminal fixture, not a task/evaluation setting.
    settings=LandingMPCSettingsV2(horizon_steps=2,terminal_horizontal_speed_mps=1e-12,
        terminal_angular_speed_rad_s=1e-12,terminal_tilt_rad=1e-12)
    old=LandingFullCommandIntervalMPCV1(interface,qp_settings=SupportSettings(),settings=settings,arm_target_rate_rad_s=10.)
    new=LandingFeasibleCommandCostMPCV1(interface,qp_settings=SupportSettings(),settings=settings,arm_target_rate_rad_s=10.)
    for tick in [0,1]:
        a=old.solve(state,state.joint_position,lane=7,force=True)
        b=new.solve(state,state.joint_position,lane=7,force=True)
        assert b['command_cost_selection']['baseline_fallback_evaluated']
        assert not b['accepted'] and not a['accepted']
        assert new._observed_tick[7]==old._observed_tick[7]==tick
        for key in ['reference_position','reference_velocity','support_anchors','previous_contacts']:
            np.testing.assert_array_equal(new._live_reference[7][key],old._live_reference[7][key])
        np.testing.assert_array_equal(b['horizon'][0]['joint_position_target'],a['horizon'][0]['joint_position_target'])
        assert new.qp.student_command_reference is None
        assert new.qp.command_change_weight==1.
        record=dict(command_cost_selection=b['command_cost_selection'],control_tick=tick,
                    accepted=b['accepted'],runtime_ms=1000*b['wall_runtime_s'])
        assert validate_selection(record,1.)=='baseline'
        # Reporting only the returned baseline time must fail: both solves
        # consumed wall time, even though only the baseline command is used.
        record['runtime_ms']=1000*b['command_cost_selection']['baseline_runtime_s']
        with pytest.raises(ValueError,match='runtime omits'):
            validate_selection(record,1.)


def test_zero_weight_does_not_launch_a_duplicate_baseline():
    _,interface,state=fixture();state.previous_command_target=state.joint_position.copy()
    c=LandingFeasibleCommandCostMPCV1(interface,qp_settings=SupportSettings(),
        settings=LandingMPCSettingsV2(horizon_steps=2),arm_target_rate_rad_s=10.,command_change_weight=0.)
    r=c.solve(state,state.joint_position,force=True,lane=0,control_tick=9)
    assert not r['command_cost_selection']['baseline_fallback_evaluated']
    assert r['command_cost_selection']['selected_weight']==0.
    record=dict(command_cost_selection=r['command_cost_selection'],control_tick=9,
                accepted=r['accepted'],runtime_ms=1000*r['wall_runtime_s'])
    assert validate_selection(record,0.)=='command_cost'
