import copy
import numpy as np
from test_admitted_arm_support_qp_v1 import fixture
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.landing_joint_sequence_proposal_v1 import knot_basis
from scripts.landing_sequence_linearization_v1 import make_nominal_profile, affine_trajectory_sequence
from scripts.condensed_implicit_drive_v1 import condense_implicit_drive


def test_zero_increment_preserves_loaded_nominal_at_each_control_boundary():
    _, interface, state = fixture()
    state.previous_command_target = state.joint_position.copy()
    saved = copy.deepcopy(state)
    c = LandingFullCommandIntervalMPCV1(interface,
        settings=LandingMPCSettingsV2(horizon_steps=4),arm_target_rate_rad_s=10.)
    plan = c.solve(state,state.joint_position,force=True)
    profile = make_nominal_profile(c,state,plan)
    assert profile is not None
    samples = list(affine_trajectory_sequence(c,state,knot_basis(4,2),profile))
    assert len(samples)==64
    for tick,node in enumerate(profile):
        end = samples[(tick+1)*16-1]
        np.testing.assert_allclose(end['velocity_bias'],node['following'].v,atol=1e-9)
        np.testing.assert_allclose(end['joint_position_bias'],node['following'].joint_position,atol=1e-9)
        np.testing.assert_allclose(end['displacement_bias'],node['displacement'],atol=1e-9)
        np.testing.assert_array_equal(end['command_bias'],node['command'])
    assert np.any(profile[0]['snapshot']['mass_matrix']!=profile[-1]['snapshot']['mass_matrix'])
    np.testing.assert_array_equal(state.q,saved.q)
    np.testing.assert_array_equal(state.v,saved.v)
    np.testing.assert_array_equal(state.previous_command_target,saved.previous_command_target)


def test_incomplete_or_failed_nominal_is_not_silently_padded():
    _, interface, state=fixture();state.previous_command_target=state.joint_position.copy()
    c=LandingFullCommandIntervalMPCV1(interface,
        settings=LandingMPCSettingsV2(horizon_steps=4),arm_target_rate_rad_s=10.)
    assert make_nominal_profile(c,state,{'horizon':[]}) is None
    assert make_nominal_profile(c,state,{'horizon':[{'accepted':True}]*3}) is None
    assert make_nominal_profile(c,state,{'horizon':[{'accepted':True}]*3+[{'accepted':False}]}) is None


def test_nonzero_knots_match_independent_time_varying_interval_condensation():
    _, interface, state=fixture();state.previous_command_target=state.joint_position.copy()
    c=LandingFullCommandIntervalMPCV1(interface,
        settings=LandingMPCSettingsV2(horizon_steps=4),arm_target_rate_rad_s=10.)
    plan=c.solve(state,state.joint_position,force=True)
    profile=make_nominal_profile(c,state,plan);assert profile is not None
    basis=knot_basis(4,2);knots=np.random.default_rng(4).normal(0,.005,(2,18))
    condensed=list(affine_trajectory_sequence(c,state,basis,profile))
    v=state.v.copy();q=state.joint_position.copy();S=np.zeros((24,18))
    S[c.qp.model.v_indices,np.arange(18)]=1.
    for tick,node in enumerate(profile):
        snap=node['snapshot'];feet=np.flatnonzero(node['state'].contacts)
        jac=snap['foot_jacobian'][feet].reshape(-1,24)
        jdot=(snap['foot_jacobian_dot'][feet]@node['state'].v).reshape(-1)
        command=node['command']+basis[tick]@knots
        direct=condense_implicit_drive(snap['mass_matrix'],snap['nonlinear'],S,
            interface.kp,interface.kd,jac,jdot,v,q,substeps=16)
        for micro,expected in enumerate(direct):
            actual=condensed[16*tick+micro]
            for name in ['velocity','joint_position','force','torque']:
                np.testing.assert_allclose(actual[name+'_map']@knots.ravel()+actual[name+'_bias'],
                    expected[name+'_map']@command+expected[name+'_bias'],atol=1e-8)
        v=direct[-1]['velocity_map']@command+direct[-1]['velocity_bias']
        q=direct[-1]['joint_position_map']@command+direct[-1]['joint_position_bias']
